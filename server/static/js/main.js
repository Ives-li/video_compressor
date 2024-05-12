Dropzone.autoDiscover = false

function toggleMode() {
  const modeSwitch = document.getElementById("mode-switch").value;
  const compressionZone = document.getElementById("compression-zone");
  const decompressionZone = document.getElementById("decompression-zone");
  const decompressionZoneMask = document.getElementById("decompression-zone-mask");
  const downloadButton = document.getElementById("download-button");
  const compressResultDisplay = document.getElementsByClassName('result-display')[0];

  // Hide all elements initially
  compressionZone.style.display = "none";
  decompressionZone.style.display = "none";
  decompressionZoneMask.style.display = "none";
  downloadButton.style.display = "none";

  // Display elements based on selected mode
  if (modeSwitch === "compression") {
    compressionZone.style.display = "block";
} else if (modeSwitch === "decompression") {
    decompressionZone.style.display = "block";
    decompressionZoneMask.style.display = "block";
    compressResultDisplay.style.display = "none";
}
}

const query = document.querySelector.bind(document);
const queryAll = document.querySelectorAll.bind(document);

const sorryMessage = 'Unfortunately, we cannot process your file...';
const errorMessages = [
  "Something seems off in this file... Hopefully, it's just a glitch! 😅",
  "Hold on! Just a moment... Are you sure about this file? 😥",
  "It seems we've got an unexpected file type... 🧐",
  "Hmm, something doesn't look right... 😳"
];
const successMessage = "The file was processed successfully. Proceed at your own risk! 😅";

function initializeDropzone() {
  let dz = new Dropzone("#compression-zone", {
    url: "http://127.0.0.1:5000/file-upload",
    paramName: "file",
    maxFiles: 1,
    dictDefaultMessage: "Drag a video here or click to upload (Max size: 500MB)",
    maxFilesize: 500,
    acceptedFiles: 'video/mp4,video/x-matroska,video/x-msvideo,video/quicktime',
    addRemoveLinks: true,
    autoProcessQueue: false,
    thumbnailWidth: 250,
    thumbnailHeight: 250
  });

  dz.on("addedfile", function () {
    if (dz.files[1] != null) {
      dz.removeFile(dz.files[0]); // Ensure only one file is in the dropzone
    }
  });

  dz.on("complete", async function (file) {
    const fileData = file.name;

    const apiEndpoint = "http://127.0.0.1:5000/compression";
  

    const response = await fetch(apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({"fileName": fileData})
    });

    const data = await response.json();


    const videoElement = query('.processed-video source');
    const downloadCompressedButton = query('.Compressed');
    const downloadMaskButton = query('.Mask');

    const resultDisplay = query('.result-display');
    const dragField = query('#compression-zone');

    // Hide the upload area and display the result area
    dragField.style.display = 'none';
    resultDisplay.style.display = 'flex';

    if (data["compressed_meas_path"]) {
      function extractFilename(path) {
        // Check for both forward and backward slashes depending on the operating system used in the path
        let parts = path.split(/[/\\]/);
        return parts[parts.length - 1];  // Get the last part, which should be the filename
      }

      // Extract filenames
      let compressedVideoFilename = extractFilename(data["compressed_meas_path"]);
      let maskZipFilename = extractFilename(data["mask_zip_path"]);

      const baseUrl = "http://127.0.0.1:5000/results/";  // Base URL where files are served
      const videoUrl = baseUrl + encodeURIComponent(compressedVideoFilename);
      const maskUrl = baseUrl + encodeURIComponent(maskZipFilename);

      videoElement.src = videoUrl;
      downloadCompressedButton.href = videoUrl;
      downloadMaskButton.href = maskUrl;

    } else {
        // Handle errors or situations where no video URL is returned
        let message = '';
        if (data['noContent'] === true) {
            message = sorryMessage;
        } else if (data['errorCount'] > 0) {
            message = errorMessages[Math.floor(Math.random() * errorMessages.length)];
        } else {
            message = successMessage;
        }
        query('.result-message').innerText = message;
    }
  });

  query(".submit").addEventListener("click", () => {
    dz.processQueue(); // Manually start processing/uploading files
  });

}

function initializeDecompressDropzone() {

  let dz = new Dropzone("#decompression-zone", {
    url: "http://127.0.0.1:5000/file-upload",
    paramName: "file",
    maxFiles: 1,
    dictDefaultMessage: "Drag a video here or click to upload (Max size: 500MB)",
    maxFilesize: 500,
    acceptedFiles: 'video/mp4,video/x-matroska,video/x-msvideo,video/quicktime',
    addRemoveLinks: true,
    autoProcessQueue: false,
    thumbnailWidth: 250,
    thumbnailHeight: 250
  });

  dz.on("addedfile", function () {
    if (dz.files[1] != null) {
      dz.removeFile(dz.files[0]); // Ensure only one file is in the dropzone
    }
  });

  dz.on("complete", async function (file) {
    const fileData = file.name;

    const apiEndpoint = "http://127.0.0.1:5000/decompression";
    // If using an ngrok URL for testing, replace the above URL with your ngrok URL

    const response = await fetch(apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({"fileName": fileData})
    });

    const data = await response.json();


    const videoElement = query('.reconstructed-video source');
    const downloadCompressedButton = query('.Reconstructed');
  

    const resultDisplay = query('.result-display-decompression');
    const dragField = query('#decompression-zone');
    const dragField_mask = query('#decompression-zone-mask')
    const compressResultDisplay = query('.result-display');


    // Hide the upload area and display the result area
    dragField.style.display = 'none';
    dragField_mask.style.display = 'none';
    compressResultDisplay.style.display = 'none';
    resultDisplay.style.display = 'flex';

    if (data["test_video_path"]) {
      function extractFilename(path) {
        // Check for both forward and backward slashes depending on the operating system used in the path
        let parts = path.split(/[/\\]/);
        return parts[parts.length - 1];  // Get the last part, which should be the filename
      }

      // Extract filenames
      let testVideoFilename = extractFilename(data["test_video_path"]);

      const baseUrl = "http://127.0.0.1:5000/results/";  // Base URL where files are served
      const videoUrl = baseUrl + encodeURIComponent(testVideoFilename);

      videoElement.src = videoUrl;
      downloadCompressedButton.href = videoUrl;

    } else {
        // Handle errors or situations where no video URL is returned
        let message = '';
        if (data['noContent'] === true) {
            message = sorryMessage;
        } else if (data['errorCount'] > 0) {
            message = errorMessages[Math.floor(Math.random() * errorMessages.length)];
        } else {
            message = successMessage;
        }
        query('.result-message').innerText = message;
    }
  });

  query("#decompression-zone-submit").addEventListener("click", () => {
    dz.processQueue(); // Manually start processing/uploading files
  });

}

window.addEventListener('DOMContentLoaded', () => {

  document.getElementById("mode-switch").addEventListener('change', toggleMode);
  initializeDropzone();
  initializeDecompressDropzone()

});