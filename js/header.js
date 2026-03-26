function appendHeader() {
  const isRoot =
    window.location.pathname === "/" ||
    window.location.pathname === "/index.html";

  const homeLink = isRoot ? "" : `<a href="/index.html">home</a>`;

  const header = document.createElement("header");
  header.innerHTML = `
    <h1>Mukesh</h1>
    <nav class="links">
      ${homeLink}
      <a href="mailto:mukeshreddy.work@gmail.com">email</a>
      <a href="https://github.com/0xmukesh">github</a>
      <a href="https://x.com/0xmukesh">twitter</a>
      <a href="/blogs">blog</a>
      <a href="/projects">projects</a>
    </nav>
  `;

  document.body.prepend(header);
}

appendHeader();
