import { useEffect, useState } from 'react';

const SESSION_STORAGE_KEY = 'rag-assistant-session';

function readSavedSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || 'null');
  } catch {
    return null;
  }
}

function saveSession(session) {
  try {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // The backend remains the source of truth if browser storage is unavailable.
  }
}

function clearSession() {
  try {
    localStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Ignore unavailable browser storage.
  }
}

function getErrorMessage(data, fallback) {
  return typeof data.detail === 'string' ? data.detail : fallback;
}

async function readResponse(response, fallback) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(getErrorMessage(data, fallback));
  }
  return data;
}

async function readEventStream(response, onEvent) {
  if (!response.body) {
    throw new Error('The server did not provide a readable answer stream.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;

  function dispatchFrame(frame) {
    const lines = frame.split('\n').map((line) => line.endsWith('\r') ? line.slice(0, -1) : line);
    const eventLine = lines.find((line) => line.startsWith('event:'));
    const dataLines = lines.filter((line) => line.startsWith('data:'));
    if (dataLines.length === 0) return;

    const eventName = eventLine ? eventLine.slice(6).trim() : 'message';
    const data = JSON.parse(dataLines.map((line) => line.slice(5).trimStart()).join('\n'));
    onEvent(eventName, data);
    if (eventName === 'done') completed = true;
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      dispatchFrame(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');
    }

    if (done) break;
  }

  if (buffer.trim()) dispatchFrame(buffer);
  if (!completed) throw new Error('The answer stream ended unexpectedly.');
}

function collectConversationSources(messages) {
  const sources = new Map();
  let currentQuestion = '';

  for (const message of messages) {
    if (message.role === 'user') {
      currentQuestion = message.content;
      continue;
    }

    for (const source of message.sources ?? []) {
      const key = `${source.page ?? 'text'}\u0000${source.text ?? ''}`;
      let entry = sources.get(key);
      if (!entry) {
        entry = { ...source, references: [] };
        sources.set(key, entry);
      }

      const reference = {
        label: source.source || 'Source',
        question: currentQuestion,
      };
      if (!entry.references.some((item) => (
        item.label === reference.label && item.question === reference.question
      ))) {
        entry.references.push(reference);
      }
    }
  }

  return [...sources.values()];
}

export default function App() {
  const [file, setFile] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [uploadStatus, setUploadStatus] = useState(null);
  const [askStatus, setAskStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const conversationSources = collectConversationSources(messages);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      const savedSession = readSavedSession();
      if (!savedSession?.documentId) {
        setRestoring(false);
        return;
      }

      setDocumentId(savedSession.documentId);
      setDocumentInfo(savedSession.documentInfo ?? null);

      if (savedSession.conversationId) {
        try {
          const response = await fetch(
            `/conversations/${encodeURIComponent(savedSession.conversationId)}`,
          );
          const data = await readResponse(response, 'Could not restore this conversation.');
          if (cancelled) return;
          setConversationId(data.conversation_id);
          setMessages(data.messages ?? []);
        } catch {
          if (cancelled) return;
          saveSession({ ...savedSession, conversationId: null });
          setConversationId(null);
          setMessages([]);
        }
      }

      if (!cancelled) setRestoring(false);
    }

    restoreSession();
    return () => { cancelled = true; };
  }, []);

  async function handleUpload(event) {
    event.preventDefault();

    if (!file) {
      setUploadStatus({ type: 'error', text: 'Please select a file.' });
      return;
    }

    const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.pdf', '.txt'].includes(extension)) {
      setUploadStatus({ type: 'error', text: 'Please upload a PDF or TXT file.' });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setUploading(true);
    setUploadStatus({
      type: 'pending',
      text: 'Uploading, extracting text and creating document chunks...',
    });
    setDocumentId(null);
    setConversationId(null);
    setDocumentInfo(null);
    setMessages([]);
    setAskStatus(null);
    clearSession();

    try {
      const response = await fetch('/upload', { method: 'POST', body: formData });
      const data = await readResponse(response, 'Upload failed. Please try again.');
      if (!data.document_id) {
        throw new Error('Upload succeeded, but the server did not return a document ID.');
      }

      setDocumentId(data.document_id);
      setDocumentInfo(data);
      saveSession({
        documentId: data.document_id,
        documentInfo: data,
        conversationId: null,
      });
      setUploadStatus({
        type: 'success',
        text: 'Document processed successfully. You can now ask questions.',
      });
    } catch (error) {
      setUploadStatus({ type: 'error', text: error.message });
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();

    if (!documentId) {
      setAskStatus({ type: 'error', text: 'Please upload a document before asking a question.' });
      return;
    }
    if (!trimmedQuestion) {
      setAskStatus({ type: 'error', text: 'Please enter a question.' });
      return;
    }

    setAsking(true);
    setAskStatus({ type: 'pending', text: 'Searching your document...' });
    setMessages((current) => [
      ...current,
      { role: 'user', content: trimmedQuestion, sources: [] },
      { role: 'assistant', content: '', sources: [] },
    ]);
    setQuestion('');

    try {
      const response = await fetch('/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: documentId,
          question: trimmedQuestion,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        await readResponse(response, 'Could not get an answer.');
      }

      await readEventStream(response, (eventName, data) => {
        if (eventName === 'conversation') {
          setConversationId(data.conversation_id);
          saveSession({
            documentId,
            documentInfo,
            conversationId: data.conversation_id,
          });
        } else if (eventName === 'sources') {
          setMessages((current) => current.map((message, index) => (
            index === current.length - 1
              ? { ...message, sources: data.sources ?? [] }
              : message
          )));
        } else if (eventName === 'token') {
          setMessages((current) => current.map((message, index) => (
            index === current.length - 1
              ? { ...message, content: message.content + (data.text ?? '') }
              : message
          )));
          setAskStatus(null);
        } else if (eventName === 'error') {
          throw new Error(data.detail || 'Could not get an answer.');
        } else if (eventName === 'done') {
          setAskStatus(null);
        }
      });
    } catch (error) {
      setAskStatus({ type: 'error', text: error.message });
      setMessages((current) => {
        const lastMessage = current[current.length - 1];
        return lastMessage?.role === 'assistant' && !lastMessage.content
          ? current.slice(0, -1)
          : current;
      });
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="container">
      <header className="header">
        <p className="eyebrow">AI Knowledge Workspace</p>
        <h1>Gemini Document Assistant</h1>
        <p className="subtitle">
          Upload a document and ask questions. Get answers grounded in your document.
        </p>
      </header>

      <section className="card" aria-labelledby="upload-heading">
        <div className="section-heading">
          <span className="step">1</span>
          <h2 id="upload-heading">Upload your document</h2>
        </div>
        <p className="help-text">Supported formats: PDF and TXT</p>
        <form onSubmit={handleUpload}>
          <label className="file-picker" htmlFor="file-input">
            <span className="file-icon" aria-hidden="true">↑</span>
            <span className="file-picker-copy">
              <strong>{file ? file.name : 'Choose a document'}</strong>
              <span>{file ? 'PDF or TXT, up to 10 MB' : 'PDF or TXT · up to 10 MB'}</span>
            </span>
            <span className="browse-label">Browse</span>
            <input
              id="file-input"
              type="file"
              accept=".txt,.pdf,text/plain,application/pdf"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              required
            />
          </label>
          <button className="primary-button" type="submit" disabled={uploading}>
            {uploading ? 'Processing document...' : 'Upload document'}
          </button>
        </form>
        {uploadStatus && <p className={`status ${uploadStatus.type}`} role="status">{uploadStatus.text}</p>}
        {documentInfo && (
          <div className="filename" role="status">
            <strong>{documentInfo.filename}</strong>
            <span>{documentInfo.chunks} chunks</span>
            <span>{documentInfo.pages_extracted} extracted pages</span>
          </div>
        )}
      </section>

      <section className="card" aria-labelledby="question-heading">
        <div className="section-heading">
          <span className="step">2</span>
          <h2 id="question-heading">Ask a question</h2>
        </div>
        <p className="help-text">Ask anything about the document you uploaded.</p>
        <form onSubmit={handleAsk}>
          <label className="visually-hidden" htmlFor="question-input">Your question</label>
          <textarea
            id="question-input"
            placeholder="Example: What is this document about? Summarize it in 20 words."
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            required
          />
          <button className="primary-button" type="submit" disabled={!documentId || asking}>
            {asking ? 'Thinking...' : 'Ask Gemini'}
          </button>
        </form>
        {askStatus && <p className={`status ${askStatus.type}`} role="status">{askStatus.text}</p>}

        {messages.length > 0 && (
          <div className="conversation-history">
            {messages.map((message, index) => (
              <article className={`message-row ${message.role}`} key={message.id ?? `${index}-${message.role}`}>
                <h3 className="message-role">{message.role === 'user' ? 'You' : 'Assistant'}</h3>
                <div className={message.role === 'user' ? 'user-message' : 'answer'}>
                  {message.content || (asking && index === messages.length - 1 ? 'Generating answer...' : 'No answer was returned.')}
                </div>
              </article>
            ))}
          </div>
        )}

        {conversationSources.length > 0 && (
          <section className="conversation-sources" aria-labelledby="sources-heading">
            <h3 id="sources-heading">Document sources</h3>
            {conversationSources.map((source, index) => (
              <article className="source-card" key={`${source.page ?? 'text'}-${index}`}>
                <h4 className="source-title">
                  {source.references.map((reference) => reference.label).join(', ')} · {source.page ? `Page ${source.page}` : 'Text file'}
                </h4>
                <p className="source-text">{source.text || ''}</p>
                <p className="source-reference">
                  Used for: {source.references.map((reference) => reference.question).filter(Boolean).join(' / ')}
                </p>
              </article>
            ))}
          </section>
        )}
      </section>

      <footer className="footer">Powered by Gemini <span aria-hidden="true">·</span> Retrieval-Augmented Generation</footer>
    </main>
  );
}