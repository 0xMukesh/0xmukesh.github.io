async function appendBlogList(sectionId, indexPath) {
  const response = await fetch(indexPath);
  const items = (await response.json())
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .filter(a => !a.draft);

  const section = document.getElementById(sectionId);
  if (!section) return;

  const isBlogs = window.location.pathname.startsWith("/blog");

  const ul = document.createElement("ul");
  ul.className = `${sectionId}-list`;
  ul.innerHTML = items.map(item => {
    const url = isBlogs ? item.url : `/blog/${item.url.split("/").pop()}`;
    return `
      <li>
        <span class="date">${item.date}</span>
        <a href="${url}">${item.title}</a>
        ${(item.tags ?? []).map(tag => `<span class="tag">#${tag}</span>`).join("")}
      </li>
    `;
  }).join("");

  section.appendChild(ul);
}
