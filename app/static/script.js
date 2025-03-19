
// functionality
function openModal(){
    const modal = document.getElementById('sideModal');
    if (modal.style.right !== '0px') {
        modal.style.right = '0px';
    } else {
        modal.style.right = '-300px';
    }
}

function checkWindowSize() {
    const smallScreenDiv = document.getElementById("myNavBar");
    const largeScreenDiv = document.getElementById("headerContainer");
    const warningScreen = document.getElementById("warningMessage");
    const sideScreen = document.getElementById("sideModal")

    if(window.innerWidth <= 800) {
        smallScreenDiv.style.display = "none";
        largeScreenDiv.style.display = "none";
        warningScreen.style.display = "block";
        sideScreen.style.display = "none";
    } else {
        warningScreen.style.display = "none"
        largeScreenDiv.style.display = "block";
        smallScreenDiv.style.display = "block";
        sideScreen.style.display = "block";
    }
}

function showLoadingScreen() {
    document.getElementById("loadingOverlay").style.display = "block";
}

function hideLoadingScreen() {
    document.getElementById("loadingOverlay").style.display = "none";
}

function toggleLoadingScreen() {
    loadingScreenStyle = document.getElementById("loadingOverlay").style.display;
    if (loadingScreenStyle == "none") {
        showLoadingScreen();
        return true;
    } else {
        hideLoadingScreen();
        return false;
    }
}


function openOverlay(text="", disallow_exit=false) {
    let overlay = document.getElementById("overlay");
    let isOpen = overlay.style.display;

    if (isOpen != "block"){
        overlay.style.display = "block";
        document.getElementById("overlay-text").innerText = "";
    } else {
        return true;
    }

    if ('finetuning_explanations' == text) {
        document.getElementById("overlay-custom").innerHTML  = "";
        document.getElementById("overlay-text").innerHTML = document.getElementById('finetuning_explanations').innerHTML;
    } 
    else if ('webdav' == text) {
        document.getElementById("overlay-custom").innerHTML = document.getElementById('webdav').innerHTML;
        document.getElementById('webdav').style.display = 'block';
    } 
    else if (typeof text !== 'string') {
        document.getElementById("overlay-custom").innerHTML = text;
    }
    else if (text.length == 0){}
    else {
        document.getElementById("overlay-text").innerText = text;
    }

    if(disallow_exit){
        overlay.onclick = "";
    } else {
        overlay.onclick = closeOverlay;
    }
}
  
function closeOverlay() {
    document.getElementById("overlay").style.display = "none";
    document.getElementById("webdav").style.display = "none";
}

function confirmAction(targetUrl) {
    let userConfirmed = null;
    switch(targetUrl) {
        case '/abort_finetune':
            userConfirmed = confirm('Are you sure you want to cancel finetuning?');
            break;
        case '/clear_loaded_data':
            userConfirmed = confirm('Are you sure you want to delete all loaded data?');
            break;
    }
   
    if (userConfirmed !== null & userConfirmed) {
        showLoadingScreen()
        fetch(targetUrl)
        .then((response)=>{
            hideLoadingScreen()
            if(response.redirected){
                window.location.href = response.url;
            }
        }) 
    }
}