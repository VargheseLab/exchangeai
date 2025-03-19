

function isValidString(input) {
    const regex = /^[a-zA-Z0-9_]+$/;
    return regex.test(input);
}

async function startFinetuning(){

    const baseModel = document.getElementById("conditionSelect").value;
    //const consent = document.getElementById("consent")?.checked;
    const modelName = document.getElementById("modelName").value;
    const trainMethod = document.getElementById("train_method").value;

    if (baseModel.match('noModel')) {
        openOverlay('Please select a base model for finetuning!')
        return
    }

    if (modelName.length < 3){
        openOverlay('Please provide a custom model name!')
        return
    }

    if (!isValidString(modelName)){
        openOverlay('Model name must only conctain Letters, Numbers and underscores!')
        return
    }

    /**
    if (consent == false){
        openOverlay('Please accept the terms!')
        return
    }
    */

    showLoadingScreen()

    const loaded_data = await fetch('/loaded_label_statistics')
    .then( resp => resp.json())
    .then( resp => {
        if (Object.keys(resp.data).length <= 0){
            return false
        }
        else return true
    })
    if (!loaded_data){
        hideLoadingScreen()
        openOverlay('Please upload data!')
        return
    }

    document.getElementById("startFinetuningButton").disabled = true;

    const lr = document.getElementById("AdvancedSettingsLR").value;
    const lr_gamma = document.getElementById("AdvancedSettingsScheduler").value;
    const batchsize = document.getElementById("AdvancedSettingsBatchSize").value;
    const epochs = document.getElementById("AdvancedSettingsEpochs").value;
    const showLogs = document.getElementById("showLogs").checked;
    

    body = new Blob(
        [JSON.stringify({
            'base_model': baseModel,
            'model_name': modelName,
            'train_method': trainMethod,
            'lr': lr,
            'lr_gamma': lr_gamma,
            'batchsize': batchsize,
            'epochs': epochs,
            'show_logs': showLogs
        })],
        {type: "application/json"}
    )
    await fetch('/run_finetune', {
        method: "POST",
        headers: { 'Content-Type': 'text/json' },
        body: body
    }).then((response)=>{         
        if(response.redirected){
            window.location.href = response.url;
        }
    }) 
}

function download_finetuned_model(model_name, url){
    fetch(url).then( response => {
            if (!response.ok) {
                return;
            }
            return response.blob()
        }
    ).then( model => {
            // Check if the Blob is empty
            if (model.size === 0) {return;}
            if (model == {}){return;}

            const hiddenElement = document.createElement('a');
            const modelURL = window.URL.createObjectURL(model);    
            hiddenElement.href = modelURL;
            hiddenElement.target = '_blank';
            hiddenElement.download = model_name + '.zip';
            hiddenElement.click();

            // Reload
            currentPath = window.location.pathname
            const pattern = /^\/finetune\/[a-f0-9\-]+$/;
            if(pattern.test(currentPath)){
                window.location.href = "/finetune"
            }
        }
    ).catch(error => {
        console.error('Error during fetch or processing:', error);
    });
}

// Export functions
function exportData() {
    const exportFormat = document.getElementById('exportFormat').value;
    const ecgType = document.getElementById('ecgType').value;

    switch (exportFormat) {
        case 'csv':
          exportToCsv(data, ecgType)
          break;

        default:
          console.log('Invalid format');
          break;
      }
    
}

function exportToCsv(data, ecgType) {
    const csvData = new Blob([data], { type: 'text/csv' });
    const csvUrl = URL.createObjectURL(csvData);
    const hiddenElement = document.createElement('a');
    hiddenElement.href = csvUrl;
    hiddenElement.target = '_blank';
    hiddenElement.download = 'exported_' + ecgType + '.csv';
    hiddenElement.click();
  }


async function uploadFiles() {
    const input = document.getElementById('ecgFileInput');
    const files = Array.from(input.files);

    const samplingRate = document.getElementById('sampling_rate').value;
    const customSamplingRate = document.getElementById('custom_sampling_rate').value;
    const adcGain = document.getElementById('adc_gain').value;
    const lead_layout = document.getElementById('lead_layout').value;

    const CHUNK_SIZE = 15;
    const totalChunks = Math.ceil(files.length / CHUNK_SIZE);

    let sends = [];

    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, files.length);
        const chunk = files.slice(start, end);

        sends.push(uploadChunkFiles(chunk, samplingRate, customSamplingRate, adcGain, lead_layout))
    }
    for (let s of sends){
        await s;
    }
    hideLoadingScreen()
    location.reload();
}

async function uploadChunkFiles(chunk, samplingRate, customSamplingRate, adcGain, lead_layout) {
    const formData = new FormData();
    for (const file of chunk) {
        formData.append('ecgFileInput', file, file.name);
    }
    formData.append('sampling_rate', samplingRate);

    if (samplingRate === 'custom') {
        formData.append('custom_sampling_rate', customSamplingRate);
    }
    formData.append('adc_gain', adcGain);
    formData.append('lead_layout', lead_layout);

    await fetch("/upload", {
        method: 'POST',
        body: formData,
    });
}
