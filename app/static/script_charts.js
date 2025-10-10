// chart functions
function updateChart(caller_value) {
    let referrer = window.location.href.split('/');
    referrer = referrer[referrer.length -1];

    let ecgSelection = null;
    let dataSourceSelect = null;

    const parentDOMs = document.getElementById('dataAndConditionSelect')?.getElementsByTagName('select')

    switch (Object.keys(caller_value)[0]) {
        case 'selectECG':
            // document is due tue templating messy, has to be searched from parentDOM
            dataSourceSelect = parentDOMs?.namedItem('dataSourceSelect')?.value;
            ecgSelection = caller_value['selectECG'];
            break;
        case 'dataSourceSelect':
            ecgSelection = document.getElementById('selectECG').value;
            dataSourceSelect = caller_value['dataSourceSelect'];
            break;
    }   

    if (referrer.match("explainable") & (!ecgSelection | !dataSourceSelect)){
        return
    }
    let predictionCondition = null

    switch (referrer) {
        case 'analyse':
            updateAnalyseChart(ecgSelection, dataSourceSelect);
            break;
        case 'predict':
            predictionCondition = parentDOMs.namedItem('conditionSelect').value;
            updatePrediction(predictionCondition);
            break;
        case 'finetune':
            break;
        case 'explainable':
            predictionCondition = parentDOMs.namedItem('conditionSelect').value;
            updatePrediction(predictionCondition);
            break;
        default:
            console.log('No charts in the refferers scope', referrer);
    }

}




async function updateAnalyseChart(ecgSelection, transform) {
    body = new Blob([JSON.stringify({'ecg': ecgSelection, 'transform': transform})], {type: "application/json"})
    const chart = await fetch('/get_ecg', {
        method: "POST",
        headers: { 'Content-Type': 'text/json' },
        body: body,
        credentials: 'include'
    }).then(response => response.json())

    if (Object.keys(chart).length == 0) {
        console.log('No data selected')
        return
    }


    config = {
        'toImageButtonOptions': {
          'format': 'svg',
          'filename': transform + '_' + ecgSelection,
          'height': null,
          'width': null
        }
    }

    Plotly.newPlot('ecg-chart', chart.data, chart.layout, config)
    
}

function plotPredictionStatistics() {
    let referrer = window.location.href.split('/');
    referrer = referrer[referrer.length -1 ];

    const parentDOMs = document.getElementById('dataAndConditionSelect')?.getElementsByTagName('select')
    const predictionCondition = parentDOMs.namedItem('conditionSelect').value;

    if (predictionCondition.match('noModel')) {
        return;
    }
    
    showLoadingScreen()

    config = {
        'toImageButtonOptions': {
          'format': 'svg',
          'filename': predictionCondition,
          'height': null,
          'width': null
        }
    }

    getPredictionStatistics(predictionCondition, true, 'distribution')
    .then( resp => {
        const predictionStatistics = document.getElementById('overlay-custom');

        if (predictionStatistics) {
            if (Object.keys(resp.data).length > 0){
                Plotly.newPlot('overlay-custom', resp.data, resp.layout, config);
            }

            hideLoadingScreen()
            openOverlay()
        }
    })
}


function plotPredictionROC() {
    let referrer = window.location.href.split('/');
    referrer = referrer[referrer.length -1 ];

    const parentDOMs = document.getElementById('dataAndConditionSelect')?.getElementsByTagName('select')
    const predictionCondition = parentDOMs.namedItem('conditionSelect').value;

    if (predictionCondition.match('noModel')) {
        return;
    }
    
    showLoadingScreen()

    const config = {
        'toImageButtonOptions': {
          'format': 'svg',
          'filename': predictionCondition,
          'height': null,
          'width': null
        }
    }

    getPredictionStatistics(predictionCondition, true, 'roc')
    .then( resp => {
        const predictionStatistics = document.getElementById('overlay-custom');
      
        if (predictionStatistics) {
            if (Object.keys(resp.data).length > 0){
                Plotly.newPlot('overlay-custom', resp.data, resp.layout, config);
            }

            hideLoadingScreen()
            openOverlay()
        }

        hideLoadingScreen()
        openOverlay()
    })
}

function downloadPredictionStatistics() {
    let referrer = window.location.href.split('/');
    referrer = referrer[referrer.length -1 ];

    const parentDOMs = document.getElementById('dataAndConditionSelect')?.getElementsByTagName('select')
    const predictionCondition = parentDOMs.namedItem('conditionSelect').value;

    if (predictionCondition.match('noModel')) {
        return;
    }

    showLoadingScreen()
    getPredictionStatistics(predictionCondition, false)
    .then( resp => {
        const csvData = new Blob([resp.export], { type: 'text/csv' });
        const csvUrl = URL.createObjectURL(csvData);
        const hiddenElement = document.createElement('a');
        hiddenElement.href = csvUrl;
        hiddenElement.target = '_blank';
        hiddenElement.download = predictionCondition + '.csv';
        hiddenElement.click();
        hideLoadingScreen()
    })
}


async function getPredictionStatistics(predictionCondition, as_plot, view='distribution'){
    const resp = await fetch('/prediction_statistics', {
        method: 'POST',
        headers: { 'Content-Type': 'text/json' },
        body: JSON.stringify({'as_plot': as_plot, 'prediction': predictionCondition, 'view': view})
    })
    .then( resp => resp.json() )

    return resp
}



// prediction
async function updatePrediction(predictionCondition) {
    let referrer = window.location.href.split('/');
    referrer = referrer[referrer.length -1 ];
    let target = '/prediction'
    let explainable = false

    if (predictionCondition.match('noModel')) {
        return;
    }

    if (referrer.match('finetune')) {
        return;
    }

    if (referrer.match('explainable')) {
        target = target + '/explainable'
        explainable = true
    }

    const ecgSelection = document.getElementById('selectECG').value;
    if (ecgSelection.match('No data')){
        return;
    }
    const predictionResultLabel = document.getElementById('predictionResultContainerLabel');
    document.getElementsByName('dataset_predictions').forEach((x) => {x.disabled = false});

    try {
        const statsResp = await fetch('/loaded_data_statistics', { method: 'GET' });
        if (statsResp.ok) {
            const stats = await statsResp.json();
            const labels_loaded = (stats && typeof stats.labels === 'number' && stats.labels > 0);
            if (labels_loaded) {
                document.getElementsByName('dataset_predictions_with_labels').forEach((x) => { x.disabled = false; });
            }
        }
    } catch (e) {
        console.warn('Failed to check loaded labels', e);
    }
    
    body = new Blob([JSON.stringify({'ecg': ecgSelection, 'prediction': predictionCondition})], {type: "application/json"})
    const prediction = await fetch(target, {
        method: "POST",
        headers: { 'Content-Type': 'text/json' },
        body: body
    }).then( response => response.json() )

    if (Object.keys(prediction).length == 0){
        return;
    }

    const predictionResultContainer = predictionResultLabel.children[0];
    if (predictionResultContainer) {
        if (!explainable) {
            predictionResultContainer.innerHTML = createTable(prediction)
        } else {
            predictionResultContainer.innerHTML = '<b>' + prediction + '</b>'
        }
    }
    
}

function createTable(prediction){
    if (prediction){
        n_elements = Object.entries(prediction).length
        var html = '<table>'
        Object.entries(prediction).forEach(function(v){
            html += '<tr><td>' + v[0] + '</td><td style="background-color:' + getColor(v[1], n_elements) + '">' + v[1] + '</td></tr>'
        });
        html += '</table>'
    }
    return html
}

function getColor(value, n_elements) {
    const threshold = (1.75 / (n_elements+1)) //Adapt to chance level!
    let lightness = ((1 + threshold - value) * 100).toString(10);
    return ["hsl(120,100%," + lightness + "%)"].join("");
}