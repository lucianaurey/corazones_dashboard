(function () {
    "use strict";

    const TINTA = "#3B2A45";
    const REJILLA = "rgba(94, 59, 115, 0.09)";
    const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
        ? "—" : Number(v).toLocaleString("es-BO", { maximumFractionDigits: d });

    document.addEventListener("DOMContentLoaded", () => {
        bindSidebar();
        bindFileInput();
        bindFormularioEvaluacion();
        bindFilasClic();
        if (window.Chart) {
            Chart.defaults.font.family = "Nunito, system-ui, sans-serif";
            Chart.defaults.color = TINTA;
            renderAll("chartsData");
            renderAll("chartsModeloData");
        }
        initTables();
    });

    // Barra lateral: en escritorio se pliega a una columna de íconos y recuerda la elección;
    // en pantallas pequeñas se abre y se cierra sobre el contenido.
    function bindSidebar() {
        const CLAVE = "corazones.sidebar";
        const movil = () => window.matchMedia("(max-width: 860px)").matches;
        const botones = document.querySelectorAll("[data-sidebar-toggle]");
        const actualizar = () => {
            const plegada = document.body.classList.contains("sidebar-collapsed");
            botones.forEach((b) => {
                b.setAttribute("aria-expanded", String(!plegada));
                b.setAttribute("title", plegada ? "Abrir panel lateral" : "Cerrar panel lateral");
            });
            window.dispatchEvent(new Event("resize"));  // los gráficos ocupan el nuevo ancho
        };
        botones.forEach((b) => b.addEventListener("click", () => {
            if (movil()) {
                document.body.classList.toggle("sidebar-open");
                return;
            }
            document.body.classList.toggle("sidebar-collapsed");
            try { localStorage.setItem(CLAVE, document.body.classList.contains("sidebar-collapsed") ? "1" : "0"); } catch (e) { /* sin almacenamiento */ }
            actualizar();
        }));
        const fondo = document.getElementById("sidebarBackdrop");
        if (fondo) fondo.addEventListener("click", () => document.body.classList.remove("sidebar-open"));
        actualizar();
        // permite la animación solo después de la primera pintura
        requestAnimationFrame(() => document.body.classList.add("sidebar-ready"));
    }

    function bindFileInput() {
        const input = document.getElementById("csv_file");
        const label = document.getElementById("nombreArchivo");
        if (!input || !label) return;
        const zone = input.closest(".dropzone");
        input.addEventListener("change", () => {
            label.textContent = input.files.length ? input.files[0].name : "Ningún archivo seleccionado";
            zone.classList.toggle("con-archivo", input.files.length > 0);
        });
        ["dragover", "dragenter"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("arrastrando"); }));
        ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.remove("arrastrando")));
        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                input.dispatchEvent(new Event("change"));
            }
        });
    }

    // Ficha de alerta: muestra los campos según el estado, el nivel del MMSE y la vista previa de la imagen
    function bindFormularioEvaluacion() {
        const form = document.querySelector(".form-evaluacion");
        if (!form) return;
        const grupos = form.querySelectorAll("[data-estados]");
        const radios = form.querySelectorAll("input[name='estado']");
        const aplicarEstado = () => {
            const actual = (form.querySelector("input[name='estado']:checked") || {}).value || "programada";
            grupos.forEach((g) => { g.hidden = !g.dataset.estados.split(" ").includes(actual); });
        };
        radios.forEach((r) => r.addEventListener("change", aplicarEstado));
        aplicarEstado();

        const puntaje = form.querySelector("input[name='mmse_puntaje']");
        const nivel = document.getElementById("nivelPreview");
        const pintarNivel = () => {
            if (!puntaje || !nivel) return;
            const v = parseInt(puntaje.value, 10);
            nivel.className = "";
            if (Number.isNaN(v) || v < 0 || v > 30) { nivel.textContent = "—"; return; }
            const k = v >= 24 ? 0 : (v >= 18 ? 1 : 2);
            nivel.textContent = ["Bajo (≥ 24)", "Medio (18–23)", "Alto (≤ 17)"][k];
            nivel.className = `badge-nivel nivel-${k}`;
        };
        if (puntaje) { puntaje.addEventListener("input", pintarNivel); pintarNivel(); }

        const input = document.getElementById("imagenResultado");
        const nombre = document.getElementById("nombreImagen");
        const vista = document.getElementById("vistaPrevia");
        const zona = document.getElementById("zonaImagen");
        if (input) {
            input.addEventListener("change", () => {
                const archivo = input.files[0];
                if (nombre) nombre.textContent = archivo ? archivo.name : "JPG, PNG, WEBP o PDF · máx. 8 MB";
                if (zona) zona.classList.toggle("con-archivo", !!archivo);
                if (vista) {
                    if (archivo && archivo.type.startsWith("image/")) {
                        vista.src = URL.createObjectURL(archivo);
                        vista.classList.remove("d-none");
                    } else {
                        vista.classList.add("d-none");
                    }
                }
            });
        }
    }

    // Filas de la tabla de alertas: un clic abre la ficha
    function bindFilasClic() {
        document.addEventListener("click", (e) => {
            const fila = e.target.closest("tr.fila-clic");
            if (!fila || e.target.closest("a, button, input, select, form")) return;
            window.location.href = fila.dataset.href;
        });
    }

    // Líneas de referencia (umbral, OR = 1, efecto mediano) sobre el eje de valores y, opcionalmente, sobre el eje X
    function trazarLinea(chart, valor, etiqueta, vertical) {
        const escala = vertical ? chart.scales.x : chart.scales.y;
        if (!escala) return;
        const pos = escala.getPixelForValue(valor);
        const { ctx, chartArea } = chart;
        ctx.save();
        ctx.strokeStyle = "#3B2A45";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        if (vertical) { ctx.moveTo(pos, chartArea.top); ctx.lineTo(pos, chartArea.bottom); }
        else { ctx.moveTo(chartArea.left, pos); ctx.lineTo(chartArea.right, pos); }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#3B2A45";
        ctx.font = "700 11px Nunito, sans-serif";
        const texto = etiqueta || `umbral ${fmt(valor, 3)}`;
        if (vertical) ctx.fillText(texto, Math.min(pos + 4, chartArea.right - ctx.measureText(texto).width - 2), chartArea.top + 10);
        else ctx.fillText(texto, chartArea.right - ctx.measureText(texto).width - 2, pos - 5);
        ctx.restore();
    }

    const lineaReferencia = {
        id: "lineaReferencia",
        afterDatasetsDraw(chart, _args, opts) {
            const horizontal = chart.options.indexAxis === "y";
            if (opts.valor !== undefined && opts.valor !== null) trazarLinea(chart, opts.valor, opts.etiqueta, horizontal);
            if (opts.valorX !== undefined && opts.valorX !== null) trazarLinea(chart, opts.valorX, opts.etiquetaX, true);
        }
    };

    function renderAll(scriptId) {
        const node = document.getElementById(scriptId);
        if (!node) return;
        let charts = [];
        try { charts = JSON.parse(node.textContent || "[]"); } catch (e) { console.error(e); }
        charts.forEach(renderChart);
    }

    function renderChart(def) {
        const canvas = document.getElementById(`chart-${def.id}`);
        if (!canvas) return;
        // Los gráficos dentro de una sección cerrada se dibujan la primera vez que se abre
        const seccion = canvas.closest("details");
        if (seccion && !seccion.open) {
            const alAbrir = () => {
                if (!seccion.open) return;
                seccion.removeEventListener("toggle", alAbrir);
                dibujar(canvas, def);
            };
            seccion.addEventListener("toggle", alAbrir);
            return;
        }
        dibujar(canvas, def);
    }

    function dibujar(canvas, def) {
        const o = def.opciones || {};
        if (def.tipo === "scatter") { renderCurva(canvas, def, o); return; }
        const circular = ["doughnut", "pie"].includes(def.tipo);
        const horizontal = !!def.horizontal;
        const datasets = def.datasets.map((d) => ({
            label: d.label,
            data: d.data,
            backgroundColor: d.colores || d.color,
            borderColor: circular ? "#fff" : (d.color || "#9D75AA"),
            borderWidth: circular ? 3 : (def.tipo === "line" ? 2.5 : 0),
            borderRadius: def.tipo === "bar" ? 6 : 0,
            pointBackgroundColor: d.color,
            pointRadius: def.tipo === "line" ? 4 : 0,
            tension: 0.3,
            fill: false,
            maxBarThickness: 46,
        }));
        const sufijo = o.porcentaje ? " %" : "";
        const ejeValor = {
            stacked: !!o.apilado,
            beginAtZero: o.min === undefined,
            min: o.min, max: o.max,
            grid: { color: REJILLA },
            ticks: { callback: (v) => `${fmt(v, 2)}${sufijo}` },
        };
        const ejeCategoria = { stacked: !!o.apilado, grid: { display: false }, ticks: { autoSkip: false, maxRotation: horizontal ? 0 : 60 } };

        new Chart(canvas, {
            type: def.tipo,
            data: { labels: def.labels, datasets },
            plugins: [lineaReferencia],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: horizontal ? "y" : "x",
                cutout: def.tipo === "doughnut" ? "62%" : undefined,
                animation: { duration: 500 },
                plugins: {
                    legend: { display: circular || datasets.length > 1, position: "bottom", labels: { usePointStyle: true, boxWidth: 10 } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const v = circular ? ctx.parsed : (horizontal ? ctx.parsed.x : ctx.parsed.y);
                                if (circular) {
                                    const total = ctx.dataset.data.reduce((a, b) => a + b, 0) || 1;
                                    return ` ${ctx.label}: ${fmt(v, 0)} (${fmt((v / total) * 100, 1)} %)`;
                                }
                                return ` ${ctx.dataset.label}: ${fmt(v, 3)}${sufijo}`;
                            }
                        }
                    },
                    lineaReferencia: { valor: o.lineaUmbral, etiqueta: o.etiquetaUmbral },
                },
                scales: circular ? {} : (horizontal ? { x: ejeValor, y: ejeCategoria } : { x: ejeCategoria, y: ejeValor }),
            },
        });
    }

    // Curvas con eje X numérico: ROC, precisión-recall y elección del umbral
    function renderCurva(canvas, def, o) {
        const datasets = def.datasets.map((d) => ({
            label: d.label,
            data: d.puntos,
            showLine: !d.soloPunto,
            borderColor: d.color,
            backgroundColor: d.relleno ? "rgba(94, 59, 115, 0.12)" : d.color,
            fill: d.relleno ? "origin" : false,
            borderWidth: d.soloPunto ? 2 : 2.4,
            borderDash: d.punteada ? [6, 5] : [],
            pointRadius: d.soloPunto ? 7 : 0,
            pointHoverRadius: d.soloPunto ? 8 : 3,
            pointBorderColor: "#fff",
            tension: 0,
        }));
        const titulo = (t) => (t ? { display: true, text: t, font: { weight: "700" } } : { display: false });
        new Chart(canvas, {
            type: "scatter",
            data: { datasets },
            plugins: [lineaReferencia],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 10 } },
                    tooltip: { callbacks: { label: (ctx) => ` ${ctx.dataset.label}: (${fmt(ctx.parsed.x, 2)}; ${fmt(ctx.parsed.y, 3)})` } },
                    lineaReferencia: { valor: o.lineaUmbral, etiqueta: o.etiquetaUmbral, valorX: o.lineaX, etiquetaX: o.etiquetaX },
                },
                scales: {
                    x: { type: "linear", min: o.xmin, max: o.xmax, grid: { color: REJILLA }, title: titulo(o.tituloX),
                         ticks: { callback: (v) => fmt(v, 2) } },
                    y: { min: o.min, max: o.max, grid: { color: REJILLA }, title: titulo(o.tituloY),
                         ticks: { callback: (v) => fmt(v, 2) } },
                },
            },
        });
    }

    function initTables() {
        if (!window.jQuery || !jQuery.fn.DataTable) return;
        jQuery("table.dt").each(function () {
            const $t = jQuery(this);
            const orden = $t.data("order");
            $t.DataTable({
                pageLength: 15,
                order: Array.isArray(orden) ? orden : [],
                dom: "<'dt-top'Bf>rt<'dt-bottom'ip>",
                buttons: [
                    { extend: "excelHtml5", text: "<i class='fa-solid fa-file-excel'></i> Excel", className: "btn btn-sm btn-outline-lila", exportOptions: { columns: ":not(.no-export)" } },
                    { extend: "csvHtml5", text: "<i class='fa-solid fa-file-csv'></i> CSV", className: "btn btn-sm btn-outline-lila", exportOptions: { columns: ":not(.no-export)" } },
                ],
                language: {
                    search: "Buscar:", lengthMenu: "Mostrar _MENU_", info: "_START_–_END_ de _TOTAL_",
                    infoEmpty: "Sin registros", zeroRecords: "Sin coincidencias", emptyTable: "Sin datos",
                    paginate: { previous: "‹", next: "›" },
                },
            });
        });
    }
})();
