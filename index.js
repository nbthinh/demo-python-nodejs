var python = require('child_process').spawn('python', ['./collaborative.py', "ABC"]);
python.stdout.on('data', function (data) {
    console.log("Python response: ", data.toString().split("\n"));
    // result.textContent = data.toString('utf8');
});