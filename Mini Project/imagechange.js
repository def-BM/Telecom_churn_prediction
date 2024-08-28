var images = ['i3.jpg', 'i5.jpg', 'i6.jpg', 'i8.jpg', 'i9.jpg', 'i10.jpg'];
var currentIndex = 0;

function changeImage() {
    currentIndex = (currentIndex + 1) % images.length;
    document.querySelector('.image').style.backgroundImage = 'url(' + images[currentIndex] + ')';
}

// Change image every 2 seconds
setInterval(changeImage, 2000);
