const tokenKey = "healthRagToken";
let token = sessionStorage.getItem(tokenKey);
let conversationId = sessionStorage.getItem("healthRagConversation");
const byId = (id) => document.getElementById(id);

function status(message) { byId("status").textContent = message; }
function authHeaders(extra = {}) { return { ...extra, Authorization: `Bearer ${token}` }; }
async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}
function showWorkspace(profile) {
  byId("auth-panel").classList.add("hidden");
  byId("workspace").classList.remove("hidden");
  byId("chat-panel").classList.remove("hidden");
  byId("welcome").textContent = `Health workspace · ${profile.full_name}`;
  loadDocuments();
  initializeConversation();
}
async function initializeConversation() {
  if (!conversationId) {
    const conversation = await request("/conversations", { method: "POST", headers: authHeaders() });
    conversationId = conversation.conversation_id;
    sessionStorage.setItem("healthRagConversation", conversationId);
    return;
  }
  try {
    const messages = await request(`/conversations/${encodeURIComponent(conversationId)}/messages`, { headers: authHeaders() });
    for (const message of messages) addBubble(message.role, message.content);
  } catch {
    conversationId = null;
    sessionStorage.removeItem("healthRagConversation");
    await initializeConversation();
  }
}
async function register() {
  const full_name = byId("name").value.trim();
  await request("/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ full_name, email: byId("email").value, password: byId("password").value }) });
  await login();
}
async function login() {
  const form = new URLSearchParams({ username: byId("email").value, password: byId("password").value });
  const result = await request("/auth/token", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form });
  token = result.access_token;
  sessionStorage.setItem(tokenKey, token);
  const profile = await request("/users/me", { headers: authHeaders() });
  showWorkspace(profile);
  status("Signed in. Your reports and conversations are scoped to this account.");
}
async function loadDocuments() {
  const docs = await request("/documents", { headers: authHeaders() });
  const list = byId("files");
  list.replaceChildren();
  for (const doc of docs) {
    const item = document.createElement("li");
    item.textContent = `${doc.source_name} · ${doc.report_type || "report"} · ${doc.report_date || "date not recorded"}`;
    list.append(item);
  }
}
function addBubble(role, text, sources = []) {
  const bubble = document.createElement("article");
  bubble.className = `bubble ${role}`;
  const content = document.createElement("div");
  content.textContent = text;
  bubble.append(content);
  if (sources.length) {
    const line = document.createElement("div");
    line.className = "sources";
    line.textContent = "Evidence: " + sources.map((source) => `${source.source_name} · chunk ${source.chunk_index} · similarity ${source.similarity}`).join("; ");
    bubble.append(line);
  }
  byId("messages").append(bubble);
  bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
async function upload() {
  const file = byId("file").files[0];
  if (!file) throw new Error("Choose a report file first.");
  const form = new FormData();
  form.append("file", file);
  const result = await request("/documents/upload", { method: "POST", headers: authHeaders(), body: form });
  addBubble("assistant", `Uploaded ${result.source_name}. Detected report type: ${result.report.report_type}. Extracted ${result.report.diagnoses.length} diagnosis items, ${result.report.lab_results.length} lab results, and ${result.report.medications.length} medications. Please compare the extracted data with your document.`);
  await loadDocuments();
  byId("file").value = "";
  status("Report analyzed and indexed.");
}
async function send() {
  const input = byId("question");
  const question = input.value.trim();
  if (!question) return;
  addBubble("user", question);
  input.value = "";
  byId("send").disabled = true;
  try {
    const result = await request("/chat", { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify({ question, conversation_id: conversationId }) });
    conversationId = result.conversation_id;
    sessionStorage.setItem("healthRagConversation", conversationId);
    addBubble("assistant", result.answer, result.retrieved_chunks);
    status("Answer grounded against your saved record context.");
  } finally { byId("send").disabled = false; }
}
byId("register").addEventListener("click", () => register().catch((error) => status(error.message)));
byId("login").addEventListener("click", () => login().catch((error) => status(error.message)));
byId("upload").addEventListener("click", () => upload().catch((error) => status(error.message)));
byId("send").addEventListener("click", () => send().catch((error) => status(error.message)));
byId("question").addEventListener("keydown", (event) => { if (event.key === "Enter") send().catch((error) => status(error.message)); });
byId("logout").addEventListener("click", () => { sessionStorage.removeItem(tokenKey); sessionStorage.removeItem("healthRagConversation"); location.reload(); });
if (token) request("/users/me", { headers: authHeaders() }).then(showWorkspace).catch(() => { sessionStorage.removeItem(tokenKey); token = null; });
