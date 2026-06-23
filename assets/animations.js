document.addEventListener("DOMContentLoaded", () => {
  const elements = document.querySelectorAll(".fade");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("show");
      }
    });
  }, {
    threshold: 0.1
  });

  elements.forEach(el => observer.observe(el));

  elements.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight) {
      el.classList.add("show");
    }
  });
});
