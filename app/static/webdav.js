const iconSVG = `<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="64" height="64" rx="8" ry="8" fill="#333333"/>
        <polyline fill="none" stroke="#c54442" stroke-width="2" stroke-miterlimit="10" points="
            0,35.54 
            15.37,35.54 
            17.79,25.96 
            20.22,35.54 
            22.95,35.54 
            24.98,43.44 
            28.6,7.02 
            31.85,50 
            33.45,35.54 
            38.68,35.54 
            41.11,31.55 
            43.97,35.54 
            59.19,35.54" 
        />
    </svg>`

let selectedFiles = []
let file_list = []
let loaded_models = []

function fillWebDav() {
  fetch("/list_files")
  .then( resp => resp.json() )
  .then( files => {
    file_list = files[0]
    loaded_models = files[1]
    
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';
    const defaultTab = 'prediction_models';

    Object.keys(file_list).forEach(tab => {
      const tabDiv = document.createElement('div');
      tabDiv.className = 'tab-content';
      tabDiv.id = `${tab}`;
      tabDiv.style.display = 'none';
    
      file_list[tab].forEach(file => {
          const div = document.createElement('div');
          div.className = 'file-item';
          div.dataset.file = file;

          const checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.value = file;

          if(tab in loaded_models){
            checkbox.checked = loaded_models[tab].includes(file);
          }
          
          const label = document.createElement('span');
          label.textContent = file.replace(/\.[^/.]+$/, ""); // Display without suffix
          
          div.appendChild(checkbox);
          div.appendChild(label);

          // Add click listener to toggle checkbox
          div.addEventListener('click', () => {
              checkbox.checked = !checkbox.checked;
          });

          tabDiv.appendChild(div);
      });
      fileList.appendChild(tabDiv);
    });

    showTab(defaultTab);
  }).then(() => {
    hideLoadingScreen()
  }).catch(error => {
    console.error('Error fetching files:', error);
    hideLoadingScreen(); // Ensure loading screen is hidden even on error
  });
}

function showTab(tab) {
  const allDivs = document.querySelectorAll('#file-list > div');

  allDivs.forEach(div => {
      div.style.display = div.id === tab ? 'block' : 'none';
  });

  // Highlight the active tab button
  const tabButtons = document.querySelectorAll('.tab-button');
  tabButtons.forEach(button => {
      button.classList.toggle('active', button.id === `${tab}-button`);
  });

}

function saveSelectedModels() {
  const allDivs = document.querySelectorAll('#file-list > div');
  let allSelectedFiles = {
    prediction_models: [],
    training_models: [],
    exchange_models: []
  };
  
  allDivs.forEach(div => {
    const checkboxes = div.querySelectorAll('.file-item input[type="checkbox"]:checked');
    checkboxes.forEach(checkbox => {
      allSelectedFiles[div.id].push(checkbox.value);
    });
  })

  fetch('/save_selected_model_list', {
      method: 'POST',
      redirect: 'follow',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ files: allSelectedFiles })
  })
  .then(response => {
    if (response.redirected) {
      window.location.href = response.url;
    }
  })
  .catch(error => {
      console.error('Error saving files:', error);
  });
}


function searchFiles() {
  const searchInput = document.getElementById('search-input').value.toLowerCase();
  const fileItems = document.querySelectorAll('.file-item');

  fileItems.forEach(item => {
    const fileName = item.dataset.file.toLowerCase();
    item.style.display = fileName.includes(searchInput) ? '' : 'none';
});
}

