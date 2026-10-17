const datasetSelect = document.getElementById('datasetSelect');
const uploadForm = document.getElementById('uploadForm');
const runBtn = document.getElementById('runBtn');
const uploadStatus = document.getElementById('uploadStatus');
const runStatus = document.getElementById('runStatus');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsContainer = document.getElementById('resultsContainer');
const emptyState = document.getElementById('emptyState');
const noQupathCheckbox = document.getElementById('noQupath');

let currentDataset = null;
let datasetsMap = {};

// Fetch Datasets on load
async function fetchDatasets() {
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        
        datasetSelect.innerHTML = '<option value="">-- Select a dataset --</option>';
        datasetsMap = {};
        
        data.datasets.forEach(ds => {
            datasetsMap[ds.id] = ds;
            const option = document.createElement('option');
            option.value = ds.id;
            option.textContent = `${ds.name} (${ds.source})`;
            datasetSelect.appendChild(option);
        });
    } catch (err) {
        console.error("Failed to fetch datasets", err);
    }
}

// Handle selection
datasetSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val && datasetsMap[val]) {
        currentDataset = datasetsMap[val];
        runBtn.disabled = false;
        uploadStatus.textContent = '';
        runStatus.textContent = '';
    } else {
        currentDataset = null;
        runBtn.disabled = true;
    }
});

// File Input UI Updates
['svs', 'json'].forEach(type => {
    const input = document.getElementById(`${type}File`);
    const label = document.getElementById(`${type}Label`);
    const drop = document.getElementById(`${type}Drop`);
    
    if (input && label && drop) {
        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                label.textContent = e.target.files[0].name;
                drop.classList.add('has-file');
            } else {
                label.textContent = 'Choose file or drag here';
                drop.classList.remove('has-file');
            }
        });
    }
});

// Handle Upload
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const svsFile = document.getElementById('svsFile').files[0];
    const jsonFile = document.getElementById('jsonFile').files[0];
    
    if (!svsFile || !jsonFile) return;
    
    const formData = new FormData();
    formData.append('svs_file', svsFile);
    formData.append('json_file', jsonFile);
    
    uploadStatus.textContent = 'Uploading... (This may take a while for large files)';
    uploadStatus.className = 'status-msg';
    
    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        if (res.ok) {
            uploadStatus.textContent = 'Upload successful!';
            uploadStatus.className = 'status-msg success';
            uploadForm.reset();
            
            // Reset custom UI state
            ['svs', 'json'].forEach(type => {
                const label = document.getElementById(`${type}Label`);
                const drop = document.getElementById(`${type}Drop`);
                if (label) label.textContent = 'Choose file or drag here';
                if (drop) drop.classList.remove('has-file');
            });
            
            // Refresh dataset list and auto-select
            await fetchDatasets();
            datasetSelect.value = data.dataset.id;
            currentDataset = data.dataset;
            datasetsMap[data.dataset.id] = data.dataset;
            runBtn.disabled = false;
        } else {
            uploadStatus.textContent = data.detail || 'Upload failed';
            uploadStatus.className = 'status-msg error';
        }
    } catch (err) {
        uploadStatus.textContent = 'Network error during upload';
        uploadStatus.className = 'status-msg error';
    }
});

// Polling Run Status
async function pollStatus(runId) {
    const res = await fetch(`/api/status/${runId}`);
    const data = await res.json();
    
    if (data.status === 'completed') {
        loadingOverlay.classList.add('hidden');
        renderResults(data.results);
        runStatus.textContent = 'Pipeline finished successfully.';
        runStatus.className = 'status-msg success';
        runBtn.disabled = false;
    } else if (data.status === 'failed') {
        loadingOverlay.classList.add('hidden');
        runStatus.textContent = 'Pipeline failed. Check console.';
        runStatus.className = 'status-msg error';
        console.error("Pipeline Error:", data.error);
        runBtn.disabled = false;
    } else {
        // Still running
        setTimeout(() => pollStatus(runId), 2000);
    }
}

// Run Pipeline
runBtn.addEventListener('click', async () => {
    if (!currentDataset) return;
    
    runBtn.disabled = true;
    runStatus.textContent = 'Starting pipeline...';
    runStatus.className = 'status-msg';
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');
    
    const formData = new FormData();
    formData.append('svs_path', currentDataset.svs_path);
    formData.append('json_path', currentDataset.json_path);
    formData.append('no_qupath', noQupathCheckbox.checked);
    
    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (res.ok) {
            pollStatus(data.run_id);
        } else {
            throw new Error('Failed to start run');
        }
    } catch (err) {
        loadingOverlay.classList.add('hidden');
        runStatus.textContent = 'Failed to trigger pipeline';
        runStatus.className = 'status-msg error';
        runBtn.disabled = false;
    }
});

// Render Results
function renderResults(results) {
    resultsContainer.classList.remove('hidden');
    
    // Images
    const imgAif = document.getElementById('imgAif');
    const imgArtery = document.getElementById('imgArtery');
    
    imgAif.src = results.aif_overlay ? (results.aif_overlay + '?t=' + new Date().getTime()) : '';
    imgAif.parentElement.style.display = results.aif_overlay ? 'block' : 'none';
    
    imgArtery.src = results.artery_overlay ? (results.artery_overlay + '?t=' + new Date().getTime()) : '';
    
    // Summary links
    const pdfLink = document.getElementById('pdfLink');
    const jsonLink = document.getElementById('jsonLink');
    
    pdfLink.href = results.pdf_summary || '#';
    pdfLink.style.display = results.pdf_summary ? 'inline-block' : 'none';
    
    jsonLink.href = results.json_summary || '#';
    jsonLink.style.display = results.json_summary ? 'inline-block' : 'none';
    
    // CSV List
    const csvList = document.getElementById('csvList');
    csvList.innerHTML = '';
    if (results.results_csv) {
        results.results_csv.forEach(csv => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = csv;
            a.target = "_blank";
            a.textContent = csv.split('/').pop();
            li.appendChild(a);
            csvList.appendChild(li);
        });
    }
}

// Tab Switching logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
    });
});

// Launch QuPath
const qupathBtn = document.getElementById('qupathBtn');
if (qupathBtn) {
    qupathBtn.addEventListener('click', async () => {
        if (!currentDataset) return;
        const formData = new FormData();
        formData.append('svs_path', currentDataset.svs_path);
        
        qupathBtn.textContent = "Launching...";
        qupathBtn.disabled = true;
        
        try {
            const res = await fetch('/api/open-qupath', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Failed to launch QuPath");
            
            setTimeout(() => {
                qupathBtn.textContent = "🔬 Open in QuPath WSI Viewer";
                qupathBtn.disabled = false;
            }, 3000);
        } catch (err) {
            alert("Error: " + err.message);
            qupathBtn.textContent = "🔬 Open in QuPath WSI Viewer";
            qupathBtn.disabled = false;
        }
    });
}

// Initialize
fetchDatasets();
