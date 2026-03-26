async function appendList(sectionId, page, jsonFilePath) {
  const response = await fetch(jsonFilePath);
  const items = (await response.json())
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .filter(a => !a.draft);

  const section = document.getElementById(sectionId);
  if (!section) return;

  const isOnPage = window.location.pathname.startsWith(`/${page}`);

  const ul = document.createElement("ul");
  ul.className = "list";
  ul.innerHTML = items.map(item => {
    const url = isOnPage ? item.url : `/${page}/${item.url.split("/").pop()}`;
    let html = `<li>`

    if (item.date) {
      html += `<span class="date">${item.date}</span>`
    }

    html += `<a href="${url}">${item.title}</a>`
    if (item.description) {
      html += `<span> - ${item.description}</span>`
    }

    html += `${(item.tags ?? []).map(tag => ` <span class="tag">#${tag}</span>`).join("")}</li>`
    return html;
  }).join("");

  section.appendChild(ul);
}
