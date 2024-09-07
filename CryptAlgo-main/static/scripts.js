function predictAlgorithm() {
    const ciphertext = document.getElementById('ciphertext').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ciphertext: ciphertext })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = `The predicted cryptographic algorithm is: ${data.algorithm}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
function clearTextarea() {
    document.getElementById('ciphertext').value = '';
    document.getElementById('result').textContent = '';
}
