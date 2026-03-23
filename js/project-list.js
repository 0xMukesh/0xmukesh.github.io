async function appendProjectList(sectionId, indexPath) {
  const response = await fetch(indexPath);
  const items = await response.json();

  const section = document.getElementById(sectionId);
  if (!section) return;

  const isProjects = window.location.pathname.startsWith("/projects");

  const ul = document.createElement("ul");
  ul.className = `${sectionId}-list`;
  ul.innerHTML = items.map(item => {
    return `
      <li>
        <a href="${item.url}">${item.title}</a> - <span>${item.description}</span>
      </li>
    `;
  }).join("");

  section.appendChild(ul);
}
