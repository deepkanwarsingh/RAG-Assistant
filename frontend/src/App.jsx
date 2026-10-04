import { useState } from 'react';

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

export default function App() {
  const [file, setFile] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [question, setQuestion] = useState('');
  const [uploadStatus, setUploadStatus] = useState(null);
  const [askStatus, setAskStatus] = useState(null);
  const [answerData, setAnswerData] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

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
    setDocumentInfo(null);
    setAnswerData(null);
    setAskStatus(null);

    try {
      const response = await fetch('/upload', { method: 'POST', body: formData });
      const data = await readResponse(response, 'Upload failed. Please try again.');
      if (!data.document_id) {
        throw new Error('Upload succeeded, but the server did not return a document ID.');
      }

      setDocumentId(data.document_id);
      setDocumentInfo(data);
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
    setAnswerData(null);

    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId, question: trimmedQuestion }),
      });
      const data = await readResponse(response, 'Could not get an answer.');
      setAnswerData(data);
      setAskStatus(null);
    } catch (error) {
      setAskStatus({ type: 'error', text: error.message });
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

        {answerData && (
          <div className="answer-section" aria-live="polite">
            <h2>Answer</h2>
            <div className="answer">{answerData.answer || 'No answer was returned.'}</div>
            {Array.isArray(answerData.sources) && answerData.sources.length > 0 && (
              <div className="sources">
                <h3>Document sources</h3>
                {answerData.sources.map((source, index) => (
                  <article className="source-card" key={`${source.source ?? index}-${index}`}>
                    <h4 className="source-title">
                      {source.source || `Source ${index + 1}`} · {source.page ? `Page ${source.page}` : 'Text file'}
                    </h4>
                    <p className="source-text">{source.text || ''}</p>
                  </article>
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      <footer className="footer">Powered by Gemini <span aria-hidden="true">·</span> Retrieval-Augmented Generation</footer>
    </main>
  );
}