document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const loading = document.getElementById('loading');
    const resultSection = document.querySelector('.result-section');

    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (previewImage) {
                        previewImage.src = e.target.result;
                        previewImage.style.display = 'block';
                    }
                }
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    // Page transition handler
    function handlePageTransition(event) {
        event.preventDefault();
        const targetUrl = event.currentTarget.href;
        
        // Get the current page content
        const pageContent = document.querySelector('.page-content');
        
        // Add slide-out animation
        pageContent.classList.add('slide-out');
        
        // Wait for animation to complete before navigating
        setTimeout(() => {
            window.location.href = targetUrl;
        }, 500);
    }

    // Add transition effect to all internal links
    const internalLinks = document.querySelectorAll('a[href^="/"]');
    internalLinks.forEach(link => {
        link.addEventListener('click', handlePageTransition);
    });

    // Add slide-in animation when page loads
    const pageContent = document.querySelector('.page-content');
    if (pageContent) {
        pageContent.classList.add('slide-in');
    }

    // Prediction form submission
    document.querySelector('form[action="/predict"]').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                // Update the prediction results on the page
                const resultsHtml = `
                    <div class="card shadow-sm mb-4">
                        <div class="card-body">
                            <h2 class="h4 text-center mb-3">Results</h2>
                            <p class="text-center mb-2">Disease Label: ${data.class_name}</p>
                            <p class="text-center text-success fw-bold">Confidence Score: ${data.confidence.toFixed(2)}%</p>
                        </div>
                    </div>`;
                document.querySelector('.prediction-results').innerHTML = resultsHtml;
            }
        })
        .catch(error => {
            showError('An error occurred while processing your request.');
        });
    });

    // Contribution form submission
    document.querySelector('form[action="/contribute"]').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        fetch('/contribute', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                // Show success message
                showSuccess(data.message);
                this.reset();
            }
        })
        .catch(error => {
            showError('An error occurred while contributing images.');
        });
    });
}); 