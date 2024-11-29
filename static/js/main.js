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

    // Your existing background image code can stay
    function setBackgroundImage(url) {
        if (!url) return;
        const img = new Image();
        img.onload = function() {
            document.body.style.backgroundImage = `url("${url}")`;
        }
        img.src = url;
    }

    // Keep your existing background fetching logic
    async function fetchNewBackground() {
        try {
            setBackgroundImage('/get_random_background');
        } catch (error) {
            console.error('Error fetching background:', error);
        }
    }

    fetchNewBackground();
    setInterval(fetchNewBackground, 30000);

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
}); 