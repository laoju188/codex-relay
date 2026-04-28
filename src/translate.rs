use serde_json::{json, Value};
use tracing::debug;

use crate::{session::SessionStore, types::*};

/// Convert a Responses API request + prior history into a Chat Completions request.
pub fn to_chat_request(
    req: &ResponsesRequest,
    history: Vec<ChatMessage>,
    sessions: &SessionStore,
) -> ChatRequest {
    let mut messages = history;

    // Prefer `instructions` (Codex CLI) over `system` (other clients).
    let system_text = req.instructions.as_ref().or(req.system.as_ref());
    if let Some(system) = system_text {
        if messages.is_empty() || messages[0].role != "system" {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: Some(system.clone()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            );
        }
    }

    // Append new input, mapping Responses API roles to Chat Completions roles.
    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(text.clone()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Messages(items) => {
            // Process items with index so we can group consecutive function_call
            // entries into a single assistant message. Providers require all tool
            // calls from one turn to live in one message with a tool_calls array.
            let mut i = 0;
            while i < items.len() {
                let item = &items[i];
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

                if item_type == "function_call" {
                    // Collect this and all immediately following function_call items
                    // into one assistant message with multiple tool_calls entries.
                    let mut grouped: Vec<Value> = Vec::new();
                    let mut call_ids: Vec<&str> = Vec::new();
                    let mut reasoning_content: Option<String> = None;

                    while i < items.len() {
                        let cur = &items[i];
                        if cur.get("type").and_then(|v| v.as_str()).unwrap_or("") != "function_call"
                        {
                            break;
                        }
                        let call_id = cur.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = cur.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let args = cur
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}");
                        call_ids.push(call_id);
                        // Try to get reasoning for EACH call_id, not just the first one.
                        // All tool_calls from the same assistant message share the same
                        // reasoning_content, so we need to find it regardless of which
                        // call_id it was stored under.
                        // Use get_reasoning_with_fallback to handle empty id case.
                        if reasoning_content.is_none() {
                            reasoning_content =
                                sessions.get_reasoning_with_fallback(call_id, name, args);
                            if let Some(ref rc) = reasoning_content {
                                debug!(
                                    call_id = %call_id,
                                    reasoning_len = rc.len(),
                                    "found reasoning_content via call_id for grouped function_calls"
                                );
                            }
                        }
                        grouped.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": { "name": name, "arguments": args }
                        }));
                        i += 1;
                    }

                    let mut msg = ChatMessage {
                        role: "assistant".into(),
                        content: None,
                        reasoning_content,
                        tool_calls: Some(grouped),
                        tool_call_id: None,
                        name: None,
                    };
                    // Fallback: try turn-level fingerprint if call_id lookup missed
                    if msg.reasoning_content.is_none() {
                        msg.reasoning_content = sessions.get_turn_reasoning(&messages, &msg);
                        if let Some(ref rc) = msg.reasoning_content {
                            debug!(
                                reasoning_len = rc.len(),
                                "found reasoning_content via turn_reasoning fallback for grouped function_calls"
                            );
                        }
                    }
                    if msg.reasoning_content.is_none() {
                        debug!(
                            call_ids = ?call_ids,
                            "WARNING: no reasoning_content found for grouped function_calls"
                        );
                    }
                    messages.push(msg);
                } else {
                    match item_type {
                        "function_call_output" => {
                            let call_id =
                                item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                            let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                            messages.push(ChatMessage {
                                role: "tool".into(),
                                content: Some(output.to_string()),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: Some(call_id.to_string()),
                                name: None,
                            });
                        }
                        _ => {
                            // Regular user/assistant/developer message
                            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                            let role = match role {
                                "developer" => "system",
                                other => other,
                            }
                            .to_string();
                            let content = value_to_text(item.get("content"));
                            let mut msg = ChatMessage {
                                role,
                                content: Some(content),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            };
                            // For assistant messages, try to recover reasoning_content
                            // from the turn-level index (needed for thinking models like
                            // DeepSeek that require reasoning_content to be passed back).
                            if msg.role == "assistant" {
                                msg.reasoning_content =
                                    sessions.get_turn_reasoning(&messages, &msg);
                            }
                            messages.push(msg);
                        }
                    }
                    i += 1;
                }
            }
        }
    }

    ChatRequest {
        model: req.model.clone(),
        messages,
        // Keep only `function` tools; providers like DeepSeek don't accept
        // OpenAI-proprietary built-ins (web_search, computer, file_search, …).
        tools: req
            .tools
            .iter()
            .filter(|t| t.get("type").and_then(Value::as_str) == Some("function"))
            .map(convert_tool)
            .collect(),
        tool_choice: req.tool_choice.clone(),
        thinking: Some(json!({ "type": "enabled" })),
        reasoning_effort: Some(map_reasoning_effort(req)),
        response_format: req.response_format.clone(),
        stream_options: if req.stream {
            Some(json!({ "include_usage": true }))
        } else {
            None
        },
        temperature: req.temperature,
        max_tokens: req.max_output_tokens,
        stream: req.stream,
    }
}

/// Map Codex/OpenAI reasoning effort to DeepSeek-compatible values.
///
/// DeepSeek only supports "high" or "max".
/// - low/medium/high → "high"
/// - xhigh/max       → "max"
fn map_reasoning_effort(req: &ResponsesRequest) -> String {
    let effort = req
        .reasoning
        .as_ref()
        .and_then(|r| r.get("effort"))
        .and_then(|v| v.as_str())
        .unwrap_or("high");

    match effort {
        "xhigh" | "max" => "max".to_string(),
        _ => "high".to_string(),
    }
}

/// Responses API tool format → Chat Completions tool format.
///
/// Responses API (flat):
///   {"type":"function","name":"foo","description":"...","parameters":{...},"strict":false}
///
/// Chat Completions (nested):
///   {"type":"function","function":{"name":"foo","description":"...","parameters":{...}}}
fn convert_tool(tool: &Value) -> Value {
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };
    // Already in Chat Completions format if it has a "function" sub-object.
    if obj.contains_key("function") {
        return tool.clone();
    }
    // Convert from Responses API flat format.
    if obj.get("type").and_then(Value::as_str) == Some("function") {
        let mut func = serde_json::Map::new();
        if let Some(v) = obj.get("name") {
            func.insert("name".into(), v.clone());
        }
        if let Some(v) = obj.get("description") {
            func.insert("description".into(), v.clone());
        }
        if let Some(v) = obj.get("parameters") {
            func.insert("parameters".into(), v.clone());
        }
        if let Some(v) = obj.get("strict") {
            func.insert("strict".into(), v.clone());
        }
        return json!({"type": "function", "function": func});
    }
    tool.clone()
}

/// Convert a Chat Completions response into a Responses API response.
pub fn from_chat_response(
    id: String,
    model: &str,
    chat: ChatResponse,
) -> (ResponsesResponse, Vec<ChatMessage>) {
    let choice = chat
        .choices
        .into_iter()
        .next()
        .unwrap_or_else(|| ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(String::new()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        });

    let text = choice.message.content.clone().unwrap_or_default();
    let usage = chat.usage.unwrap_or(ChatUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    let response = ResponsesResponse {
        id,
        object: "response",
        model: model.to_string(),
        output: vec![ResponsesOutputItem {
            kind: "message".into(),
            role: "assistant".into(),
            content: vec![ContentPart {
                kind: "output_text".into(),
                text: Some(text),
            }],
        }],
        usage: ResponsesUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        },
    };

    (response, vec![choice.message])
}

/// Collapse a Responses API content value (string or parts array) to plain text.
fn value_to_text(v: Option<&Value>) -> String {
    match v {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base_req(input: ResponsesInput) -> ResponsesRequest {
        ResponsesRequest {
            model: "test".into(),
            input,
            previous_response_id: None,
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
            tool_choice: None,
            response_format: None,
            parallel_tool_calls: None,
            store: None,
            metadata: None,
        }
    }

    #[test]
    fn test_text_input_becomes_user_message() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Text("hello".into()));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].content.as_deref(), Some("hello"));
    }

    #[test]
    fn test_system_prompt_from_instructions() {
        let sessions = SessionStore::new();
        let mut req = base_req(ResponsesInput::Text("hi".into()));
        req.instructions = Some("be helpful".into());
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].content.as_deref(), Some("be helpful"));
    }

    #[test]
    fn test_developer_role_mapped_to_system() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "developer", "content": "secret instructions"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content.as_deref(),
            Some("secret instructions")
        );
    }

    #[test]
    fn test_function_call_grouping() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "c1", "name": "fn_a", "arguments": "{}"}),
            json!({"type": "function_call", "call_id": "c2", "name": "fn_b", "arguments": "{}"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "c1");
        assert_eq!(calls[1]["id"], "c2");
    }

    #[test]
    fn test_function_call_output_becomes_tool_message() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call_output", "call_id": "c1", "output": "result"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].content.as_deref(), Some("result"));
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn test_convert_tool_flat_to_nested() {
        let flat = json!({
            "type": "function",
            "name": "my_fn",
            "description": "does stuff",
            "parameters": {"type": "object"}
        });
        let nested = convert_tool(&flat);
        assert_eq!(nested["type"], "function");
        assert_eq!(nested["function"]["name"], "my_fn");
        assert_eq!(nested["function"]["description"], "does stuff");
    }

    #[test]
    fn test_convert_tool_already_nested() {
        let already = json!({
            "type": "function",
            "function": {"name": "my_fn", "description": "does stuff"}
        });
        let result = convert_tool(&already);
        assert_eq!(result, already);
    }

    #[test]
    fn test_value_to_text_string() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": "plain text"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].content.as_deref(), Some("plain text"));
    }

    #[test]
    fn test_value_to_text_parts_array() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hello "},
                {"type": "input_text", "text": "world"}
            ]}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].content.as_deref(), Some("hello world"));
    }

    // ── Reasoning Content Tests ─────────────────────────────────────────────────

    /// Test: reasoning_content is recovered when function_call is replayed
    /// with call_id lookup.
    #[test]
    fn test_reasoning_recovered_via_call_id_lookup() {
        let sessions = SessionStore::new();

        // Simulate: reasoning was stored under call_id "call_ds_1"
        sessions.store_reasoning(
            "call_ds_1".into(),
            "<think>step by step thinking</think>".into(),
        );

        // Codex replays the function_call with the same call_id
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "call_ds_1", "name": "exec", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_ds_1", "output": "done"}),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);

        // Should have 2 messages: assistant (with reasoning) + tool
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(
            chat.messages[0].reasoning_content.as_deref(),
            Some("<think>step by step thinking</think>"),
            "assistant should have reasoning_content from call_id lookup"
        );
        assert!(chat.messages[0].tool_calls.is_some());
        assert_eq!(chat.messages[1].role, "tool");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_ds_1"));
    }

    /// Test: when first function_call has no reasoning but second one does,
    /// reasoning should still be found.
    #[test]
    fn test_reasoning_found_via_second_call_id() {
        let sessions = SessionStore::new();

        // Reasoning was stored under call_id "call_ds_2"
        sessions.store_reasoning(
            "call_ds_2".into(),
            "<think>analysis for second call</think>".into(),
        );

        // Codex replays two function_calls, first without reasoning, second with
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "call_ds_1", "name": "noop", "arguments": "{}"}),
            json!({"type": "function_call", "call_id": "call_ds_2", "name": "exec", "arguments": "{}"}),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);

        // Single grouped assistant message with both tool_calls
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(
            chat.messages[0].reasoning_content.as_deref(),
            Some("<think>analysis for second call</think>"),
            "reasoning should be found via second call_id"
        );
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
    }

    /// Test: reasoning is NOT present for non-thinking models
    #[test]
    fn test_no_reasoning_for_non_thinking_model() {
        let sessions = SessionStore::new();
        // Don't store any reasoning - simulating non-thinking model

        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "exec", "arguments": "{}"}),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        assert!(
            chat.messages[0].reasoning_content.is_none(),
            "no reasoning_content for non-thinking model"
        );
    }

    /// Test: assistant message with content and tool_calls gets reasoning via
    /// turn_reasoning fallback when call_id lookup fails.
    #[test]
    fn test_reasoning_via_turn_reasoning_fallback() {
        let sessions = SessionStore::new();

        // Store reasoning via turn_reasoning (content-based key)
        let assistant = ChatMessage {
            role: "assistant".into(),
            content: Some("Let me check".into()),
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "exec", "arguments": "{}"}
            })]),
            tool_call_id: None,
            name: None,
        };
        sessions.store_turn_reasoning(&[], &assistant, "<think>thinking...</think>".into());

        // Replay without previous_response_id - Codex sends content+function_call separately
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check"}]}),
            json!({"type": "function_call", "call_id": "call_xyz", "name": "exec", "arguments": "{}"}),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);

        // Should find reasoning via turn_reasoning fallback
        assert_eq!(chat.messages.len(), 2);
        // function_call item's grouped assistant message should have reasoning
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(
            chat.messages[0].reasoning_content.as_deref(),
            Some("<think>thinking...</think>"),
            "reasoning should be found via turn_reasoning fallback"
        );
    }

    /// Test: multiple function_call_outputs replayed after one assistant message
    #[test]
    fn test_multiple_function_call_outputs_replay() {
        let sessions = SessionStore::new();

        sessions.store_reasoning(
            "call_1".into(),
            "<think>parallel thinking for call 1 and 2</think>".into(),
        );

        // Two function_calls from the same assistant turn
        // Two tool outputs in the replay
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "func1", "arguments": "{}"}),
            json!({"type": "function_call", "call_id": "call_2", "name": "func2", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "result1"}),
            json!({"type": "function_call_output", "call_id": "call_2", "output": "result2"}),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);

        // Should be: 1 assistant (with reasoning + 2 tool_calls) + 2 tool messages
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(
            chat.messages[0].reasoning_content.as_deref(),
            Some("<think>parallel thinking for call 1 and 2</think>"),
            "reasoning should be shared across both function_calls"
        );
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        // Tool messages in order
        assert_eq!(chat.messages[1].role, "tool");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[2].role, "tool");
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_2"));
    }

    /// Test: reasoning effort mapping — low/medium/high → "high"
    #[test]
    fn test_reasoning_effort_high() {
        let sessions = SessionStore::new();
        for effort in &["low", "medium", "high", "unknown"] {
            let req = ResponsesRequest {
                model: "test".into(),
                input: ResponsesInput::Text("hi".into()),
                previous_response_id: None,
                tools: vec![],
                stream: false,
                temperature: None,
                max_output_tokens: None,
                system: None,
                instructions: None,
                reasoning: Some(json!({ "effort": effort })),
                tool_choice: None,
                response_format: None,
                parallel_tool_calls: None,
                store: None,
                metadata: None,
            };
            let chat = to_chat_request(&req, vec![], &sessions);
            assert_eq!(
                chat.reasoning_effort.as_deref(),
                Some("high"),
                "effort={effort} should map to high"
            );
            assert_eq!(
                chat.thinking.as_deref(),
                Some(&json!({"type": "enabled"})),
                "thinking should always be enabled"
            );
        }
    }

    /// Test: reasoning effort mapping — xhigh/max → "max"
    #[test]
    fn test_reasoning_effort_max() {
        let sessions = SessionStore::new();
        for effort in &["xhigh", "max"] {
            let req = ResponsesRequest {
                model: "test".into(),
                input: ResponsesInput::Text("hi".into()),
                previous_response_id: None,
                tools: vec![],
                stream: false,
                temperature: None,
                max_output_tokens: None,
                system: None,
                instructions: None,
                reasoning: Some(json!({ "effort": effort })),
                tool_choice: None,
                response_format: None,
                parallel_tool_calls: None,
                store: None,
                metadata: None,
            };
            let chat = to_chat_request(&req, vec![], &sessions);
            assert_eq!(
                chat.reasoning_effort.as_deref(),
                Some("max"),
                "effort={effort} should map to max"
            );
        }
    }

    /// Test: reasoning effort defaults to "high" when not specified
    #[test]
    fn test_reasoning_effort_default() {
        let sessions = SessionStore::new();
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("hi".into()),
            previous_response_id: None,
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
            tool_choice: None,
            response_format: None,
            parallel_tool_calls: None,
            store: None,
            metadata: None,
        };
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.reasoning_effort.as_deref(), Some("high"));
    }

    /// Test: tool_choice is transparently passed through
    #[test]
    fn test_tool_choice_pass_through() {
        let sessions = SessionStore::new();
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("hi".into()),
            previous_response_id: None,
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
            tool_choice: Some(json!("auto")),
            response_format: Some(json!({"type": "json_object"})),
            parallel_tool_calls: None,
            store: None,
            metadata: None,
        };
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.tool_choice.as_deref(), Some(&json!("auto")));
        assert_eq!(
            chat.response_format.as_deref(),
            Some(&json!({"type": "json_object"}))
        );
    }

    /// Test: stream_options includes include_usage when streaming
    #[test]
    fn test_stream_options_include_usage() {
        let sessions = SessionStore::new();
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("hi".into()),
            previous_response_id: None,
            tools: vec![],
            stream: true,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
            tool_choice: None,
            response_format: None,
            parallel_tool_calls: None,
            store: None,
            metadata: None,
        };
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(
            chat.stream_options.as_deref(),
            Some(&json!({"include_usage": true}))
        );
    }

    /// Test: stream_options is None when not streaming
    #[test]
    fn test_stream_options_none_when_not_streaming() {
        let sessions = SessionStore::new();
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("hi".into()),
            previous_response_id: None,
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
            tool_choice: None,
            response_format: None,
            parallel_tool_calls: None,
            store: None,
            metadata: None,
        };
        let chat = to_chat_request(&req, vec![], &sessions);
        assert!(chat.stream_options.is_none());
    }
}
