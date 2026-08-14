async function appendBadges(sectionId, jsonFilePath) {
  const response = await fetch(jsonFilePath);
  const items = await response.json();

  const section = document.getElementById(sectionId);
  if (!section) return;

  const div = document.createElement("div");
  div.className = "badges";
  div.innerHTML = items
    .map((item) => {
      const img = `<img src="${item.src}" alt="${item.name} badge" />`;
      return item.href ? `<a href="${item.href}">${img}</a>` : img;
    })
    .join("");

  section.appendChild(div);
}
