async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to fetch ${path}: ${response.ok}`);
  }

  return response.json();
}

// used for scaffolding the header
// usage: <header data-component="header"></header>
function renderHeader(el) {
  el.innerHTML = `
    <h1>Mukesh</h1>
    <nav class="links">
        <a href="/index.html">home</a>
        <a href="/work.html">work</a>
        <a href="/blog">blog</a>
        <a href="/projects">projects</a>
        <a href="/contact.html">contact</a>
    </nav>`;
}

// used for rendering list of items from a json file
// json file path is passed via `data-json` attribute
// usage: <section data-component="list" data-json="..."></section>
async function renderList(el) {
  const jsonPath = el.dataset.json;
  const base = new URL(jsonPath, window.location.href).href;

  const data = await loadJSON(jsonPath);
  const items = data
    .filter((item) => !item.draft)
    .sort((a, b) => new Date(b.date) - new Date(a.date));

  const ul = document.createElement("ul");
  ul.className = "list";
  ul.innerHTML = items
    .map((item) => {
      const url = new URL(item.url, base).href;
      const date = item.date ? `<span class="date">${item.date}</span>` : "";
      const description = item.description
        ? `<span> - ${item.description}</span>`
        : "";
      const tags = (item.tags ?? [])
        .map((tag) => `<span class="tag">#${tag}</span>`)
        .join(" ");

      return `<li>
        ${date}
        <a href="${url}">${item.title}</a> ${tags}${description}
      </li>`;
    })
    .join("");

  el.appendChild(ul);
}

// used for rendering 88x31 badges from a json file
// json file path is passed via `data-json` attribute
// usage: <div data-component="badges" data-json="..."></div>
async function renderBadges(el) {
  const items = await loadJSON(el.dataset.json);

  const div = document.createElement("div");
  div.className = "badges";
  div.innerHTML = items
    .map((item) => {
      const img = `<img src="${item.src}" alt="${item.name} badge" />`;
      return item.href ? `<a href="${item.href}">${img}</a>` : img;
    })
    .join("");

  el.appendChild(div);
}

// used to render blog header
// usage: <div data-component="blog-header" data-json="..."></div>
async function renderBlogHeader(el) {
  const jsonPath = el.dataset.json;
  const slug = window.location.pathname.split("/").pop();
  const data = await loadJSON(jsonPath);

  const post = data.find((p) => p.url.endsWith(slug));
  if (!post) {
    return;
  }

  const tags = (post.tags ?? [])
    .map((tag) => `<span class="tag">#${tag}</span>`)
    .join("");

  el.innerHTML = `
    <nav class="blog-nav">
      <a href="/blog">← Go back</a>
      <a href="/index.html">Home</a>
    </nav>
    <div class="blog-header">
      <h1>${post.title}</h1>
      <p class="blog-meta">${post.date}</p>
      ${tags}
    </div>`;
}

const components = {
  header: renderHeader,
  list: renderList,
  badges: renderBadges,
  "blog-header": renderBlogHeader,
};

// finds all elements with `data-component` attribute and then processes them
function renderComponents() {
  document.querySelectorAll("[data-component]").forEach((el) => {
    const render = components[el.dataset.component];
    if (!render) {
      throw new Error(`unknown component: ${el.dataset.component}`);
    }

    Promise.resolve(render(el)).catch((error) => {
      throw new Error(`could not render ${el.dataset.component}: ${error}`);
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderComponents);
} else {
  renderComponents();
}
