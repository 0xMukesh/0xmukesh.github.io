function appendFooter() {
  const footer = document.createElement("footer");
  footer.innerHTML = `
    <p>© ${new Date().getFullYear()} Mukesh · This website uses <a href="../vendor/latex-css">LaTeX.css</a> · <a href="mailto:mukesh@0xc84.fyi">mukesh@0xc84.fyi</a></p>
  `;

  document.body.appendChild(footer);
}

appendFooter();
