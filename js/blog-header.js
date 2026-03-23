async function appendBlogHeader() {
  const slug = window.location.pathname
    .split("/")
    .filter(Boolean)
    .pop()

  const response = await fetch("/blog/index.json");
  const posts = await response.json();

  const post = posts.find(p => p.url.endsWith(slug));
  if (!post) return;

  const nav = document.createElement("nav");
  nav.className = "blog-nav";
  nav.innerHTML = `
    <a href="/blog">← Go back</a>
    <a href="/index.html">Home</a>
  `;

  const header = document.createElement("div");
  header.className = "blog-header";
  header.innerHTML = `
    <h1>${post.title}</h1>
    <p class="blog-meta">${post.date}</p>
    ${(post.tags ?? []).map(tag => `<span class="tag">#${tag}</span>`).join("")}
  `;

  document.body.prepend(header);
  document.body.prepend(nav);
}

appendBlogHeader()
