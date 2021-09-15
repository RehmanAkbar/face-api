const express = require("express");
const fileUpload = require("express-fileupload");
const faceApiService = require("./faceApiService");
var cors = require('cors')
const app = express();
const port = process.env.PORT || 3000;

app.use(fileUpload());
app.use(cors());

app.post("/upload", async (req, res) => {

  if (!req.files)
    return res.status(400).json({ message: 'No files were uploaded.' });

  var file = req.files.reference;
  
  const result = await faceApiService.recognition(req.files);
  // return result;
  // const result = await faceApiService.detect(file.data, file.name);

  res.json({
    result
    // url: `http://localhost:3000/out/${file.name}`,
  });
});

app.use("/out", express.static("out"));

app.listen(port, () => {
  console.log("Server started on port" + port);
});
