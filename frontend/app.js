const pages = {
    dashboard: "Dashboard",
    chunking: "Chunking",
    embedding: "Embedding",
    rag: "Query / RAG"
};

function showPage(page) {

    document.querySelectorAll(".page").forEach(el => {
        el.classList.remove("active");
    });

    const targetPage = document.getElementById(page + "-page");

    if (targetPage) {
        targetPage.classList.add("active");
    }

    document.querySelectorAll(".nav-item").forEach(el => {
        el.classList.remove("active");
    });

    document.querySelectorAll(`[data-page="${page}"]`).forEach(el => {
        el.classList.add("active");
    });

    const title = document.getElementById("page-title");

    if (title) {
        title.textContent = pages[page] || page;
    }
}


document.addEventListener("click", event => {

    const button = event.target.closest("[data-page]");

    if (!button) return;

    showPage(button.dataset.page);
});


function format(value) {

    if (value === undefined || value === null) {
        return "-";
    }

    const number = Number(value);

    if (Number.isNaN(number)) {
        return value;
    }

    return number.toFixed(3);
}


function escapeHtml(text) {

    return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


async function loadChunking() {

    try {

        const response = await fetch("/api/chunking");
        const data = await response.json();

        const container = document.getElementById("chunking-content");

        const rows = Array.isArray(data)
            ? data
            : data.results || data.methods || [];

        if (!rows.length) {

            container.innerHTML = `
                <div class="panel">
                    <p class="panel-text">
                        Aucun résultat de chunking exploitable.
                    </p>
                </div>
            `;

            return;
        }

        container.innerHTML = `
            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>MÉTHODE</th>
                            <th>CHUNKS</th>
                            <th>P@5</th>
                            <th>R@5</th>
                            <th>F1@5</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${rows.map((row, index) => `
                            <tr class="${index === 0 ? "best-row" : ""}">
                                <td class="method-name">
                                    ${escapeHtml(row.method || row.name || "-")}
                                </td>
                                <td>${row.chunks ?? "-"}</td>
                                <td>${format(row.precision_at_5 ?? row.P_at_5)}</td>
                                <td>${format(row.recall_at_5 ?? row.R_at_5)}</td>
                                <td>${format(row.f1_at_5 ?? row.F1_at_5)}</td>
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        `;

    } catch (error) {

        console.error("Erreur chunking :", error);
    }
}


async function loadEmbedding() {

    try {

        const response = await fetch("/api/embedding");
        const data = await response.json();

        const container = document.getElementById("embedding-content");

        const rows = Array.isArray(data)
            ? data
            : data.results || data.methods || [];

        if (!rows.length) {

            container.innerHTML = `
                <div class="panel">
                    <p class="panel-text">
                        Aucun résultat d'embedding exploitable.
                    </p>
                </div>
            `;

            return;
        }

        container.innerHTML = `
            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>MÉTHODE</th>
                            <th>DIM</th>
                            <th>INTRA</th>
                            <th>INTER</th>
                            <th>P@5</th>
                            <th>R@5</th>
                            <th>F1@5</th>
                            <th>MAP@5</th>
                            <th>MRR@5</th>
                            <th>NDCG@5</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${rows.map((row, index) => `
                            <tr class="${index === 0 ? "best-row" : ""}">
                                <td class="method-name">
                                    ${escapeHtml(row.method || row.name || "-")}
                                </td>
                                <td>${row.dim ?? row.dimension ?? "-"}</td>
                                <td>${format(row.intra)}</td>
                                <td>${format(row.inter)}</td>
                                <td>${format(row.P_at_5 ?? row.precision_at_5)}</td>
                                <td>${format(row.R_at_5 ?? row.recall_at_5)}</td>
                                <td>${format(row.F1_at_5 ?? row.f1_at_5)}</td>
                                <td>${format(row.MAP_at_5 ?? row.map_at_5)}</td>
                                <td>${format(row.MRR_at_5 ?? row.mrr_at_5)}</td>
                                <td>${format(row.NDCG_at_5 ?? row.ndcg_at_5)}</td>
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        `;

    } catch (error) {

        console.error("Erreur embedding :", error);
    }
}


async function searchDocument() {

    const queryInput = document.getElementById("query");
    const container = document.getElementById("query-results");

    const query = queryInput.value.trim();

    if (!query) {

        container.innerHTML = `
            <div class="result">
                <div class="result-text">
                    Veuillez saisir une question.
                </div>
            </div>
        `;

        return;
    }

    container.innerHTML = `
        <div class="result">
            <div class="result-score">
                RAG · QWEN 2.5:3B
            </div>

            <div class="result-text">
                Recherche des passages pertinents et génération de la réponse...
            </div>
        </div>
    `;

    try {

        const response = await fetch("/api/query", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query,
                top_k: 5
            })
        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail
                    ? JSON.stringify(data.detail)
                    : "Erreur API"
            );
        }


        const answer = data.answer || "Aucune réponse générée.";
        const sources = Array.isArray(data.sources)
            ? data.sources
            : [];


        container.innerHTML = `

            <div class="rag-answer panel">

                <div class="panel-header">

                    <div>
                        <span class="panel-kicker">
                            RÉPONSE GÉNÉRÉE
                        </span>

                        <h3>
                            Qwen 2.5 · RAG
                        </h3>
                    </div>

                    <span class="rank">
                        ${escapeHtml(data.model || "qwen2.5:3b")}
                    </span>

                </div>


                <div class="answer-text">
                    ${escapeHtml(answer).replaceAll("\n", "<br>")}
                </div>


                <div class="rag-meta">

                    <span>
                        Embedding :
                        <strong>
                            ${escapeHtml(data.embedding || "Sentence-BERT")}
                        </strong>
                    </span>

                    <span>
                        Chunking :
                        <strong>
                            ${escapeHtml(data.chunking || "Fixed Token")}
                        </strong>
                    </span>

                    <span>
                        Top-K :
                        <strong>
                            ${data.top_k ?? 5}
                        </strong>
                    </span>

                </div>

            </div>


            <div class="sources-heading">

                <div class="eyebrow">
                    RÉCUPÉRATION
                </div>

                <h3>
                    Sources utilisées
                </h3>

                <p>
                    Les passages ci-dessous ont été transmis au modèle
                    pour générer la réponse.
                </p>

            </div>


            <div class="sources-list">

                ${
                    sources.length
                        ? sources.map((source, index) => `

                            <div class="result source-card">

                                <div class="result-score">

                                    SOURCE #${source.rank ?? index + 1}

                                    · Score

                                    ${Number(source.score ?? 0).toFixed(4)}

                                </div>

                                <div class="result-text">
                                    ${escapeHtml(source.text)}
                                </div>

                            </div>

                        `).join("")
                        : `
                            <div class="result">
                                <div class="result-text">
                                    Aucune source récupérée.
                                </div>
                            </div>
                        `
                }

            </div>
        `;

    } catch (error) {

        console.error("Erreur RAG :", error);

        container.innerHTML = `

            <div class="result">

                <div class="result-score">
                    ERREUR RAG
                </div>

                <div class="result-text">
                    Impossible de contacter le pipeline RAG.
                    Vérifiez que le backend FastAPI et Ollama sont démarrés.
                </div>

            </div>

        `;
    }
}


document.getElementById("search-btn")
    .addEventListener("click", searchDocument);


document.getElementById("query")
    .addEventListener("keydown", event => {

        if (event.key === "Enter" && !event.shiftKey) {

            event.preventDefault();

            searchDocument();
        }
    });


loadChunking();
loadEmbedding();
