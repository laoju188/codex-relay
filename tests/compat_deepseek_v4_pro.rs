//! Vendor compatibility tests for reasoning_content round-trip.
//!
//! These tests simulate the exact request patterns observed from Codex CLI
//! when talking to DeepSeek V4 Pro (and similar thinking models) through the
//! relay. The key behavior: Codex CLI sends `previous_response_id: None` and
//! embeds all conversation history as `input` items. The relay must recover
//! `reasoning_content` and attach it to the corresponding assistant messages.

use codex_relay::session::SessionStore;
use codex_relay::translate::to_chat_request;
use codex_relay::types::*;
use serde_json::json;

fn base_req(input: ResponsesInput) -> ResponsesRequest {
    ResponsesRequest {
        model: "deepseek-v4-pro".into(),
        input,
        previous_response_id: None,
        tools: vec![],
        stream: false,
        temperature: None,
        max_output_tokens: None,
        system: None,
        instructions: None,
    }
}

fn assistant_msg(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".into(),
        content: Some(content.into()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn assistant_msg_with_tool_calls(content: &str, tool_calls: Vec<serde_json::Value>) -> ChatMessage {
    ChatMessage {
        role: "assistant".into(),
        content: Some(content.into()),
        reasoning_content: None,
        tool_calls: Some(tool_calls),
        tool_call_id: None,
        name: None,
    }
}

/// DeepSeek V4 Pro: 2-turn text-only conversation where turn 1 produces
/// reasoning_content that must be recovered in turn 2.
#[test]
fn test_deepseek_v4_pro_reasoning_roundtrip_text_only() {
    let store = SessionStore::new();

    // Simulate turn 1: model returned text + reasoning
    let assistant = assistant_msg("Let me analyze this");
    store.store_turn_reasoning(
        &[],
        &assistant,
        "<think>analyzing the problem...</think>".into(),
    );

    // Turn 2: Codex replays full conversation history as input items
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Research task prompt"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me analyze this"}]}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user "Research task prompt"
    // messages[1] = assistant "Let me analyze this"  ← should have reasoning
    // messages[2] = user "Continue"
    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(
        chat.messages[1].content.as_deref(),
        Some("Let me analyze this")
    );
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>analyzing the problem...</think>"),
        "assistant text message should have reasoning_content recovered"
    );
}

/// DeepSeek V4 Pro: turn 1 returns text + reasoning + tool_calls. Codex
/// replays them as SEPARATE items (assistant message + function_call items +
/// function_call_output items). Both the text message and the grouped
/// tool-call message should get reasoning_content.
#[test]
fn test_deepseek_v4_pro_reasoning_roundtrip_with_tool_calls() {
    let store = SessionStore::new();

    // Simulate turn 1: model returned text + tool_calls + reasoning
    let assistant = assistant_msg_with_tool_calls(
        "Let me check",
        vec![json!({
            "id": "call_abc",
            "type": "function",
            "function": {"name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}
        })],
    );
    store.store_turn_reasoning(&[], &assistant, "<think>need to read files</think>".into());

    // Turn 2: Codex replays conversation with separate items
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Prompt"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check"}]}),
        json!({"type": "function_call", "call_id": "call_abc", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}),
        json!({"type": "function_call_output", "call_id": "call_abc", "output": "file1.py\nfile2.py"}),
        json!({"type": "message", "role": "user", "content": "What next?"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user "Prompt"
    // messages[1] = assistant "Let me check"        ← reasoning via content key
    // messages[2] = assistant tool_calls [call_abc]  ← reasoning via call_id fallback
    // messages[3] = tool output
    // messages[4] = user "What next?"
    assert_eq!(chat.messages.len(), 5);

    // Assistant TEXT message should have reasoning_content
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Let me check"));
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>need to read files</think>"),
        "assistant text message should have reasoning_content"
    );

    // Assistant TOOL CALL message should also have reasoning_content (via call_id)
    assert_eq!(chat.messages[2].role, "assistant");
    assert!(chat.messages[2].tool_calls.is_some());
    assert_eq!(
        chat.messages[2].reasoning_content.as_deref(),
        Some("<think>need to read files</think>"),
        "assistant tool-call message should have reasoning_content via call_id fallback"
    );
}

/// DeepSeek V4 Pro: 3-turn conversation where each turn has its own reasoning
/// that must be independently recovered.
#[test]
fn test_deepseek_v4_pro_multi_turn_reasoning() {
    let store = SessionStore::new();

    // Store reasoning for turn 1
    let assistant1 = assistant_msg("Step 1 analysis");
    store.store_turn_reasoning(
        &[],
        &assistant1,
        "<think>first pass thinking</think>".into(),
    );

    // Store reasoning for turn 2
    let assistant2 = assistant_msg("Step 2 deeper look");
    store.store_turn_reasoning(
        &[],
        &assistant2,
        "<think>second pass thinking</think>".into(),
    );

    // Turn 3: Codex replays the full 2-turn history
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Start research"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Step 1 analysis"}]}),
        json!({"type": "message", "role": "user", "content": "Go deeper"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Step 2 deeper look"}]}),
        json!({"type": "message", "role": "user", "content": "Finalize"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    assert_eq!(chat.messages.len(), 5);

    // Turn 1 assistant
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Step 1 analysis"));
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>first pass thinking</think>"),
        "turn 1 assistant should have its own reasoning_content"
    );

    // Turn 2 assistant
    assert_eq!(chat.messages[3].role, "assistant");
    assert_eq!(
        chat.messages[3].content.as_deref(),
        Some("Step 2 deeper look")
    );
    assert_eq!(
        chat.messages[3].reasoning_content.as_deref(),
        Some("<think>second pass thinking</think>"),
        "turn 2 assistant should have its own reasoning_content"
    );
}

/// Non-thinking model (e.g. deepseek-chat): when no reasoning was stored,
/// assistant messages should have reasoning_content=None.
#[test]
fn test_non_thinking_model_no_reasoning_content() {
    let store = SessionStore::new();

    // Don't store any reasoning — simulating a model that doesn't think

    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Hello"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi there!"}]}),
        json!({"type": "message", "role": "user", "content": "Thanks"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Hi there!"));
    assert!(
        chat.messages[1].reasoning_content.is_none(),
        "non-thinking model should have reasoning_content=None"
    );
}

/// Kimi K2.6: verify call_id-based reasoning recovery still works when Codex
/// DOES use previous_response_id and the function_call path is the main
/// recovery mechanism.
#[test]
fn test_kimi_k2_6_reasoning_via_call_id() {
    let store = SessionStore::new();

    // Store reasoning keyed by call_id (the existing mechanism)
    store.store_reasoning("call_xyz".into(), "<think>kimi is thinking</think>".into());

    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Do something"}),
        json!({"type": "function_call", "call_id": "call_xyz", "name": "run_cmd", "arguments": "{\"cmd\": \"pwd\"}"}),
        json!({"type": "function_call_output", "call_id": "call_xyz", "output": "/home/user"}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user
    // messages[1] = assistant (grouped function_call)  ← reasoning via call_id
    // messages[2] = tool output
    // messages[3] = user
    assert_eq!(chat.messages.len(), 4);
    assert_eq!(chat.messages[1].role, "assistant");
    assert!(chat.messages[1].tool_calls.is_some());
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>kimi is thinking</think>"),
        "grouped assistant tool-call message should have reasoning_content via call_id"
    );
}

/// Test A (GPT suggestion): Grouped function_calls with 2-3 call_ids.
/// Using only one call_id to query should find the same assistant record and reasoning_content.
#[test]
fn test_grouped_function_calls_any_call_id_finds_reasoning() {
    let store = SessionStore::new();

    // Store reasoning for call_id "call_2"
    store.store_reasoning(
        "call_2".into(),
        "<think>shared reasoning for all calls</think>".into(),
    );

    // Codex replays 3 grouped function_calls, only referencing "call_2"
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Do three things"}),
        json!({"type": "function_call", "call_id": "call_1", "name": "noop", "arguments": "{}"}),
        json!({"type": "function_call", "call_id": "call_2", "name": "exec", "arguments": "{\"x\":1}"}),
        json!({"type": "function_call", "call_id": "call_3", "name": "noop2", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_2", "output": "done"}),
        json!({"type": "function_call_output", "call_id": "call_3", "output": "ok"}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user
    // messages[1] = assistant (grouped 3 function_calls)  ← reasoning via call_2
    // messages[2,3,4] = tool outputs
    // messages[5] = user
    assert_eq!(chat.messages.len(), 6);
    assert_eq!(chat.messages[1].role, "assistant");
    assert!(chat.messages[1].tool_calls.is_some());
    assert_eq!(chat.messages[1].tool_calls.as_ref().unwrap().len(), 3);
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>shared reasoning for all calls</think>"),
        "reasoning should be found via any of the grouped call_ids"
    );
}

/// Test B (GPT suggestion): First turn DeepSeek stream returns reasoning_content + tool_calls.
/// Second turn Codex replays function_call_output. Verify outgoing ChatRequest's
/// assistant message includes reasoning_content.
#[test]
fn test_turn2_replay_with_reasoning_from_turn1() {
    let store = SessionStore::new();

    // Turn 1: DeepSeek returned reasoning + tool_calls
    // We store reasoning under the call_id
    store.store_reasoning(
        "call_t1".into(),
        "<think>first turn thinking about the task</think>".into(),
    );

    // Turn 2: Codex replays the function_call and its output
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Initial request"}),
        json!({"type": "function_call", "call_id": "call_t1", "name": "tool1", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_t1", "output": "result"}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user
    // messages[1] = assistant (grouped function_call)  ← should have reasoning
    // messages[2] = tool output
    // messages[3] = user
    assert_eq!(chat.messages.len(), 4);
    assert_eq!(chat.messages[1].role, "assistant");
    assert!(chat.messages[1].tool_calls.is_some());
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>first turn thinking about the task</think>"),
        "assistant should have reasoning_content from turn 1"
    );
}

/// Test C (GPT suggestion): DeepSeek V4 Flash style where tool_call.id is empty.
/// Reasoning is stored under fallback key (name + arguments hash).
/// Codex replays with the same name+arguments, lookup should find reasoning.
#[test]
fn test_empty_id_stores_and_finds_via_fallback_key() {
    let store = SessionStore::new();

    // Simulate DeepSeek returning tool_call with empty id
    // We stored reasoning under fallback key because id was empty
    // The fallback key is computed as: fallback_{hash("tool_name::" + arguments)}
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    "tool_exec::".hash(&mut hasher);
    "{\"cmd\":\"ls\"}".hash(&mut hasher);
    let fallback_key = format!("fallback_{}", hasher.finish());
    store.store_reasoning(
        fallback_key,
        "<think>thinking about exec command</think>".into(),
    );

    // Codex replays with same name and arguments (but call_id might be different or empty)
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Run command"}),
        json!({"type": "function_call", "call_id": "call_replay_1", "name": "tool_exec", "arguments": "{\"cmd\":\"ls\"}"}),
        json!({"type": "function_call_output", "call_id": "call_replay_1", "output": "files"}),
        json!({"type": "message", "role": "user", "content": "Done"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    assert_eq!(chat.messages.len(), 4);
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>thinking about exec command</think>"),
        "reasoning should be found via fallback key when call_id is different"
    );
}

/// Test D (GPT suggestion): Without previous_response_id, full input_items replay
/// with mixed function_call groups (some with reasoning, some without).
/// All should be correctly processed.
#[test]
fn test_full_history_replay_mixed_reasoning() {
    let store = SessionStore::new();

    // Store reasoning for turn 1
    store.store_reasoning("call_1".into(), "<think>turn 1 reasoning</think>".into());

    // Store reasoning for turn 2 via fallback
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    "tool_turn2::".hash(&mut hasher);
    "{\"a\":1}".hash(&mut hasher);
    let fk2 = format!("fallback_{}", hasher.finish());
    store.store_reasoning(fk2, "<think>turn 2 reasoning via fallback</think>".into());

    // Full history replay with no previous_response_id
    let req = base_req(ResponsesInput::Messages(vec![
        // Turn 1
        json!({"type": "message", "role": "user", "content": "First request"}),
        json!({"type": "function_call", "call_id": "call_1", "name": "tool_turn1", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_1", "output": "result1"}),
        // Turn 2
        json!({"type": "function_call", "call_id": "call_replay_2", "name": "tool_turn2", "arguments": "{\"a\":1}"}),
        json!({"type": "function_call_output", "call_id": "call_replay_2", "output": "result2"}),
        // Turn 3
        json!({"type": "message", "role": "user", "content": "Third request"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages: user, assistant(call_1), tool, assistant(call_replay_2), tool, user
    assert_eq!(chat.messages.len(), 6);
    assert_eq!(chat.messages[0].role, "user");
    assert_eq!(chat.messages[1].role, "assistant");
    assert!(chat.messages[1].tool_calls.is_some());
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>turn 1 reasoning</think>")
    );
    assert_eq!(chat.messages[3].role, "assistant");
    assert!(chat.messages[3].tool_calls.is_some());
    assert_eq!(
        chat.messages[3].reasoning_content.as_deref(),
        Some("<think>turn 2 reasoning via fallback</think>")
    );
}
