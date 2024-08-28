
// Changing image
var images = ['i3.jpg', 'i8.jpg', 'i5.jpg', 'i7.jpg', 'i6.jpg', 'i4.jpg'];
var currentIndex = 0;

function changeImage() {
    currentIndex = (currentIndex + 1) % images.length;
    document.querySelector('.image').style.backgroundImage = 'url(' + images[currentIndex] + ')';
}

// Change image every 5 seconds
setInterval(changeImage, 5000);
