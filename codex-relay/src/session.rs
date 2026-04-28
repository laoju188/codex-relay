use dashmap::DashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use tracing::debug;
use uuid::Uuid;

use crate::types::ChatMessage;

/// Generate a fallback reasoning key when call_id is empty.
/// Uses name + arguments hash to create a reproducible key.
pub fn fallback_reasoning_key(name: &str, arguments: &str) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{}::", name).hash(&mut hasher);
    arguments.hash(&mut hasher);
    format!("fallback_{}", hasher.finish())
}

/// Maps response_id → accumulated message history for that session.
/// Codex uses `previous_response_id` to continue a conversation; we maintain
/// the full messages[] here so each Chat Completions call is self-contained.
///
/// Also maintains call_id → reasoning_content so that thinking-capable models
/// (e.g. kimi-k2.6) can have their reasoning_content round-tripped back when
/// Codex replays tool-call history in subsequent requests.
///
/// For assistant messages without tool calls (pure text), reasoning_content
/// is indexed by a fingerprint of the prior messages + assistant content,
/// so it can be recovered when Codex replays the full conversation in `input`
/// without using `previous_response_id`.
#[derive(Clone)]
pub struct SessionStore {
    inner: Arc<DashMap<String, Vec<ChatMessage>>>,
    reasoning: Arc<DashMap<String, String>>,
    /// fingerprint(prior_messages, assistant_content) → reasoning_content
    turn_reasoning: Arc<DashMap<u64, String>>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
            reasoning: Arc::new(DashMap::new()),
            turn_reasoning: Arc::new(DashMap::new()),
        }
    }

    /// Store reasoning_content keyed by the tool call_id so it can be
    /// injected back when the same call_id appears in a subsequent request.
    pub fn store_reasoning(&self, call_id: String, reasoning: String) {
        if !reasoning.is_empty() {
            let reasoning_len = reasoning.len();
            self.reasoning.insert(call_id.clone(), reasoning);
            debug!(
                call_id = %call_id,
                reasoning_len = reasoning_len,
                "stored reasoning_content for call_id"
            );
        }
    }

    /// Look up stored reasoning_content for a call_id.
    pub fn get_reasoning(&self, call_id: &str) -> Option<String> {
        self.reasoning.get(call_id).map(|v| v.clone())
    }

    /// Look up reasoning with fallback: first try call_id directly, then try
    /// fallback_reasoning_key(name, arguments) if call_id is empty or not found.
    /// This handles the case where DeepSeek returns tool_calls with empty id and
    /// Codex generates its own call_id for replay.
    pub fn get_reasoning_with_fallback(
        &self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Option<String> {
        // First try direct call_id lookup
        if let Some(r) = self.get_reasoning(call_id) {
            return Some(r);
        }
        // Fallback: if call_id is empty or direct lookup failed, try name+arguments key.
        // This is needed when DeepSeek returned empty id (stored under fallback key) but
        // Codex generates its own call_id for replay.
        let fk = fallback_reasoning_key(name, arguments);
        debug!(
            fallback_key = %fk,
            call_id = %call_id,
            name = %name,
            "get_reasoning_with_fallback: trying fallback key after direct lookup failed"
        );
        if let Some(r) = self.get_reasoning(&fk) {
            return Some(r);
        }
        None
    }

    /// Store reasoning_content for an assistant turn, keyed by a fingerprint
    /// of the assistant message content and tool calls.
    ///
    /// When content is empty but tool_calls exist, we still need to store
    /// the reasoning so it can be recovered when function_call items are
    /// replayed in subsequent requests.
    pub fn store_turn_reasoning(
        &self,
        _prior: &[ChatMessage],
        assistant: &ChatMessage,
        reasoning: String,
    ) {
        if !reasoning.is_empty() {
            // Store under content-only key so lookups work even when Codex
            // replays the assistant text and function_calls as separate items.
            // Only store if content is non-empty, otherwise the content key
            // would be the same for all empty-content messages.
            let content = assistant.content.as_deref().unwrap_or("");
            if !content.is_empty() {
                let key = Self::content_key(content);
                self.turn_reasoning.insert(key, reasoning.clone());
            }
            // Also store under each tool call_id (existing mechanism).
            // This is the CRITICAL path for reasoning without text content -
            // when content is empty but tool_calls exist, reasoning is only
            // stored via this path.
            if let Some(tcs) = &assistant.tool_calls {
                for tc in tcs {
                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        if !id.is_empty() {
                            self.store_reasoning(id.to_string(), reasoning.clone());
                        }
                    }
                }
            }
        }
    }

    /// Look up reasoning_content for an assistant turn by its text content.
    /// Falls back to call_id lookup when content is empty (for function_call items
    /// where content is None but reasoning was stored via tool call ids).
    pub fn get_turn_reasoning(
        &self,
        _prior: &[ChatMessage],
        assistant: &ChatMessage,
    ) -> Option<String> {
        let content = assistant.content.as_deref().unwrap_or("");
        // Try content-based key first if content is non-empty
        if !content.is_empty() {
            let key = Self::content_key(content);
            if let Some(reasoning) = self.turn_reasoning.get(&key) {
                return Some(reasoning.clone());
            }
        }
        // Fallback: try call_id lookup when content is empty or content-based lookup missed.
        // This is needed because reasoning is stored via call_id when content is empty
        // (e.g., for thinking models with tool_calls but no text content).
        if let Some(tcs) = &assistant.tool_calls {
            for tc in tcs {
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        if let Some(reasoning) = self.get_reasoning(id) {
                            return Some(reasoning);
                        }
                    }
                }
            }
        }
        None
    }

    /// Hash assistant message content for turn-level reasoning lookup.
    fn content_key(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Retrieve history for a prior response_id, or empty vec if not found.
    pub fn get_history(&self, response_id: &str) -> Vec<ChatMessage> {
        self.inner
            .get(response_id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Allocate a fresh response_id without storing anything yet.
    /// Use with save_with_id() for the streaming path.
    pub fn new_id(&self) -> String {
        format!("resp_{}", Uuid::new_v4().simple())
    }

    /// Store under a pre-allocated response_id (streaming path).
    pub fn save_with_id(&self, id: String, messages: Vec<ChatMessage>) {
        self.inner.insert(id, messages);
    }

    /// Allocate an id and store atomically (non-streaming path).
    pub fn save(&self, messages: Vec<ChatMessage>) -> String {
        let id = self.new_id();
        self.inner.insert(id.clone(), messages);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    fn msg(role: &str, content: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.map(Into::into),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn test_store_and_get_reasoning() {
        let store = SessionStore::new();
        store.store_reasoning("call_1".into(), "think".into());
        assert_eq!(store.get_reasoning("call_1"), Some("think".into()));
    }

    #[test]
    fn test_get_reasoning_missing() {
        let store = SessionStore::new();
        assert_eq!(store.get_reasoning("nonexistent"), None);
    }

    #[test]
    fn test_empty_reasoning_not_stored() {
        let store = SessionStore::new();
        store.store_reasoning("call_e".into(), "".into());
        assert_eq!(store.get_reasoning("call_e"), None);
    }

    #[test]
    fn test_turn_reasoning_by_content() {
        let store = SessionStore::new();
        let assistant = msg("assistant", Some("hello world"));
        store.store_turn_reasoning(&[], &assistant, "deep thought".into());
        assert_eq!(
            store.get_turn_reasoning(&[], &assistant),
            Some("deep thought".into())
        );
    }

    #[test]
    fn test_turn_reasoning_empty_content() {
        let store = SessionStore::new();
        let assistant = msg("assistant", Some(""));
        store.store_turn_reasoning(&[], &assistant, "reason".into());
        assert_eq!(store.get_turn_reasoning(&[], &assistant), None);
    }

    #[test]
    fn test_turn_reasoning_also_stores_call_ids() {
        let store = SessionStore::new();
        let mut assistant = msg("assistant", Some("hi"));
        assistant.tool_calls = Some(vec![serde_json::json!({
            "id": "call_123",
            "type": "function",
            "function": {"name": "exec", "arguments": "{}"}
        })]);
        store.store_turn_reasoning(&[], &assistant, "reason_tc".into());
        assert_eq!(store.get_reasoning("call_123"), Some("reason_tc".into()));
    }

    #[test]
    fn test_history_save_and_get() {
        let store = SessionStore::new();
        let msgs = vec![msg("user", Some("hi")), msg("assistant", Some("hey"))];
        let id = store.save(msgs.clone());
        let got = store.get_history(&id);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].content.as_deref(), Some("hi"));

        // save_with_id
        let id2 = store.new_id();
        store.save_with_id(id2.clone(), vec![msg("user", Some("q"))]);
        assert_eq!(store.get_history(&id2).len(), 1);
    }

    #[test]
    fn test_content_key_deterministic() {
        let a = SessionStore::content_key("same text");
        let b = SessionStore::content_key("same text");
        assert_eq!(a, b);
        let c = SessionStore::content_key("different");
        assert_ne!(a, c);
    }
}
