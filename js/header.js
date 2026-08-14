function appendHeader() {
  const isRoot =
    window.location.pathname === "/" ||
    window.location.pathname === "/index.html";

  const homeLink = isRoot ? "" : `<a href="/index.html">home</a>`;

  const header = document.createElement("header");
  header.innerHTML = `
    <h1>Mukesh</h1>
    <nav class="links">
      <a href="/index.html">home</a>
      <a href="/blog">blog</a>
      <a href="/projects">projects</a>
      <a href="/contact.html">contact</a>
    </nav>
  `;

  document.body.prepend(header);
}

appendHeader();
