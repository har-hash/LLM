document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');
    const queryCard = document.getElementById('query-card');
    const questionInput = document.getElementById('question-input');
    const queryButton = document.getElementById('query-button');
    const resultsCard = document.getElementById('results-card');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('results-content');

    // --- State ---
    let sessionId = null;
    const API_BASE_URL = 'http://127.0.0.1:8000';

    // --- Event Listeners ---
    uploadButton.addEventListener('click', handleUpload);
    queryButton.addEventListener('click', handleQuery);

    // --- Functions ---
    async function handleUpload() {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'red';
            return;
        }

        // Generate a simple unique session ID for this upload
        sessionId = `session_${Date.now()}`;
        uploadStatus.textContent = 'Uploading and processing...';
        uploadStatus.style.color = 'orange';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);

        try {
            const response = await fetch(`${API_BASE_URL}/upload_document/`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const data = await response.json();
            uploadStatus.textContent = data.message;
            uploadStatus.style.color = 'green';
            queryCard.style.display = 'block'; // Show the query section
            questionInput.focus();

        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.style.color = 'red';
        }
    }

    async function handleQuery() {
        const question = questionInput.value.trim();
        if (!question) {
            alert('Please enter a question.');
            return;
        }
        if (!sessionId) {
            alert('Please upload a document first.');
            return;
        }

        resultsCard.style.display = 'block';
        loader.style.display = 'block';
        resultsContent.innerHTML = ''; // Clear previous results

        const requestBody = {
            session_id: sessionId,
            question: question,
        };

        try {
            const response = await fetch(`${API_BASE_URL}/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Query failed');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            resultsContent.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
        } finally {
            loader.style.display = 'none';
        }
    }

    function displayResults(data) {
        let decisionClass = 'Information';
        if (data.decision.toLowerCase().includes('approved') || data.decision.toLowerCase().includes('covered')) {
            decisionClass = 'Approved';
        } else if (data.decision.toLowerCase().includes('rejected') || data.decision.toLowerCase().includes('not')) {
            decisionClass = 'Rejected';
        }

        let html = `
            <div class="decision ${decisionClass}">${data.decision}</div>
            <h3>Justification</h3>
            <p>${data.justification}</p>
        `;

        if (data.conditions) {
            html += `
                <h3>Conditions</h3>
                <p>${data.conditions}</p>
            `;
        }

        if (data.referenced_clauses && data.referenced_clauses.length > 0) {
            html += '<h3>Referenced Clauses</h3>';
            data.referenced_clauses.forEach(clause => {
                html += `
                    <div class="referenced-clause">
                        <strong>Clause ${clause.clause_number || 'N/A'} (from ${clause.document_name})</strong>
                        <p>"${clause.text}"</p>
                    </div>
                `;
            });
        }
        resultsContent.innerHTML = html;
    }
});