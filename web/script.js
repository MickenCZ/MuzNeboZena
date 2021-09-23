const result = document.getElementById("result");
const classes = {0: "Muž", 1: "Žena"}
const fileInput = document.getElementById("fileInput")
const imageTag = document.getElementById("imageTag")
const submitButton = document.getElementById("submit")
const submitURL = document.getElementById("submitURL")
const URLInput = document.getElementById("url")
const pTag = document.getElementById("p")

imageTag.addEventListener("click", () => {
  if (pTag.style.display == "none") {
    pTag.style.display = "block"
  }
  else {
    pTag.style.display = "none"
  }
})

fileInput.addEventListener("change", () => {
  let reader = new FileReader();
  reader.onload = () => {
    let dataURL = reader.result
    imageTag.src = dataURL
    imageTag.style.display = "block"
    result.innerHTML = ""
  }
  let file = fileInput.files[0]
  reader.readAsDataURL(file)
})

const model = await tf.loadLayersModel("./JsModel7Epoch/model.json")

URLInput.addEventListener("change", () => {
  try {imageTag.src = URLInput.value
    imageTag.style.display = "block"
    result.innerHTML = ""
    imageTag.crossOrigin = "Anonymous"
  }
    catch (err) {
      imageTag.style.display = "none"
    }
})

submitButton.addEventListener("click", async () => {
  let image = imageTag 
  let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([180,180]).toFloat().expandDims()

  result.innerHTML = "načítání..."
  let predictions = await model.predict(tensor).data()
  let classification = Array.from(predictions).map((p,i) => {
    return {probability:p, className: classes[i]}
  }).sort((a,b) => {
    return b.probability - a.probability
  }).slice(0, 5)
  
result.innerHTML = ""
pTag.innerHTML = ""

classification.forEach(p => { 
  pTag.append(p.className + " : " + p.probability.toFixed(0) + "\n") 
})

if (classification[0].probability > classification[1].probability) {
    result.append("Je to " + classification[0].className)
  }
  else {
    result.append("Je to " + classification[1].className)
  }
})


submitURL.addEventListener("click", async () => {
  URLInput.value = ""
  imageTag.style.display = "block"

  let image = imageTag 
  let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([180,180]).toFloat().expandDims()

 

  result.innerHTML = "načítání..."
  let predictions = await model.predict(tensor).data()
  let classification = Array.from(predictions).map((p,i) => {
    return {probability:p, className: classes[i]}
  }).sort((a,b) => {
    return b.probability - a.probability
  }).slice(0, 5)

  result.innerHTML = ""
  pTag.innerHTML = ""

  classification.forEach(p => { 
    pTag.append(p.className + " : " + p.probability.toFixed(0) + "\n") 
  })

  if (classification[0].probability > classification[1].probability) {
    result.append("Je to " + classification[0].className)
  }
  else {
    result.append("Je to " + classification[1].className)
  }
})
