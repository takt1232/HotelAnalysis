function updateFileName(input) {
  var fileName = input.files[0].name;
  var label = document.getElementById("fileLabel");
  label.innerText = fileName;
}

function validateForm() {
  var fileInput = document.getElementById("inputCSVFile");
  if (fileInput.files.length === 0) {
    alert("Please select a file.");
    return false; // Prevent form submission
  }
  return true; // Allow form submission
}
