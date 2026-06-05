function appendFooter() {
  const footer = document.createElement("footer");
  footer.innerHTML = `
    <p>© ${new Date().getFullYear()} Mukesh · This website was inspired from <a href="../vendor/latex-css">LaTeX.css</a> · <a href="mailto:mukeshreddy.work@gmail.com">mukeshreddy.work@gmail.com</a></p>
  `;

  document.body.appendChild(footer);
}

appendFooter();
