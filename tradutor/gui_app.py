"""Interface desktop do tradutor com a identidade visual da NinjaSoft."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from .config import AppConfig, load_config
from .gui import (
    TranslationJob,
    build_cli_command,
    expected_output_paths,
    required_api_key_environment,
    validate_translation_job,
)
from .languages import LANGUAGES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).resolve().parent / "assets"

ColorValue = str | tuple[str, str]

COLORS: dict[str, ColorValue] = {
    "bg": ("#F4F4F6", "#0B0D20"),
    "bg_deep": ("#FFFFFF", "#060710"),
    "surface": ("#FFFFFF", "#121631"),
    "surface_strong": ("#ECEAF8", "#1A1F42"),
    "border": ("#DAD8E8", "#272D59"),
    "text": ("#17162A", "#F5F4FF"),
    "muted": ("#66657A", "#B9BBD1"),
    "accent": ("#6941C6", "#9D72EF"),
    "accent_hover": ("#7B55D1", "#B993FF"),
    "accent_text": ("#FFFFFF", "#0B0D20"),
    "blue": "#4E61C4",
    "success": ("#178B62", "#55D6A3"),
    "danger": ("#C23B55", "#EF7C8E"),
    "log_text": ("#303047", "#D7DAF0"),
}

BACKEND_LABELS = {
    "Ollama · local": "ollama",
    "Google Gemini · API": "gemini",
    "OpenAI · API": "openai",
}
BACKEND_CODES = {value: label for label, value in BACKEND_LABELS.items()}
DESQUEBRAR_LABELS = {
    "LLM · melhor acabamento": "llm",
    "Seguro · sem chamada extra": "safe",
}


class TranslatorApp(ctk.CTk):
    """Janela principal para configurar, executar e acompanhar traduções."""

    def __init__(self) -> None:
        """Carrega a configuração local e constrói a janela principal."""
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=COLORS["bg"])
        self.config_data = self._load_project_config()
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.cancel_requested = False
        self.active_job: TranslationJob | None = None
        self.backend_models = {
            "ollama": self.config_data.translate_model,
            "gemini": "",
            "openai": "",
        }

        self.title("Ninja Translator · NinjaSoft")
        self.geometry("1180x820")
        self.minsize(1020, 720)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_variables()
        self._build_sidebar()
        self._build_workspace()
        self._refresh_backend_fields()
        self.after(120, self._drain_message_queue)

    @staticmethod
    def _load_project_config() -> AppConfig:
        """Lê a configuração do repositório, mesmo quando a GUI parte de outra pasta."""
        config_path = PROJECT_ROOT / "config.yaml"
        return load_config(config_path if config_path.exists() else None)

    def _create_variables(self) -> None:
        """Centraliza o estado editável dos controles da interface."""
        self.language_options = ["Detecção automática"]
        self.language_options.extend(
            f"{language.name_pt.capitalize()} · {language.code}" for language in LANGUAGES
        )
        self.language_codes = {"Detecção automática": "auto"}
        self.language_codes.update(
            {
                f"{language.name_pt.capitalize()} · {language.code}": language.code
                for language in LANGUAGES
            }
        )

        configured_language = self.config_data.source_language
        selected_language = next(
            (label for label, code in self.language_codes.items() if code == configured_language),
            "Detecção automática",
        )
        self.input_var = ctk.StringVar()
        self.language_var = ctk.StringVar(value=selected_language)
        self.backend_var = ctk.StringVar(
            value=BACKEND_CODES.get(self.config_data.translate_backend, "Ollama · local")
        )
        self.model_var = ctk.StringVar(value=self.config_data.translate_model)
        self.api_key_var = ctk.StringVar()
        self.timeout_var = ctk.StringVar(value=str(self.config_data.request_timeout))
        self.glossary_var = ctk.StringVar()
        self.use_glossary_var = ctk.BooleanVar(value=False)
        self.refine_var = ctk.BooleanVar(value=self.config_data.refine_after_translate)
        self.repair_var = ctk.BooleanVar(value=self.config_data.use_translation_repair)
        self.pdf_var = ctk.BooleanVar(value=self.config_data.pdf_enabled)
        self.resume_var = ctk.BooleanVar(value=False)
        self.debug_var = ctk.BooleanVar(value=False)
        self.theme_var = ctk.StringVar(value="dark")
        self.desquebrar_var = ctk.StringVar(value="LLM · melhor acabamento")
        if self.config_data.desquebrar_mode == "safe":
            self.desquebrar_var.set("Seguro · sem chamada extra")

    def _build_sidebar(self) -> None:
        """Monta a faixa de marca e o resumo do fluxo na lateral."""
        sidebar = ctk.CTkFrame(self, width=252, corner_radius=0, fg_color=COLORS["bg_deep"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(8, weight=1)

        dark_logo_path = ASSET_DIR / "ninjasoft-logo.png"
        light_logo_path = ASSET_DIR / "ninjasoft-logo-light.png"
        if dark_logo_path.exists():
            dark_logo = Image.open(dark_logo_path)
            light_logo = Image.open(light_logo_path) if light_logo_path.exists() else dark_logo
            display_width = 178
            display_height = max(
                32,
                min(72, round(dark_logo.height * display_width / dark_logo.width)),
            )
            self.logo_image = ctk.CTkImage(
                light_image=light_logo,
                dark_image=dark_logo,
                size=(display_width, display_height),
            )
            ctk.CTkLabel(sidebar, text="", image=self.logo_image).grid(
                row=0, column=0, padx=30, pady=(34, 8), sticky="w"
            )
        else:
            ctk.CTkLabel(
                sidebar,
                text="NINJASOFT",
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color=COLORS["text"],
            ).grid(row=0, column=0, padx=30, pady=(36, 8), sticky="w")

        ctk.CTkLabel(
            sidebar,
            text="NINJA TRANSLATOR",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["accent"],
        ).grid(row=1, column=0, padx=31, pady=(0, 36), sticky="w")

        ctk.CTkLabel(
            sidebar,
            text="FLUXO DE TRABALHO",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["muted"],
        ).grid(row=2, column=0, padx=31, pady=(0, 12), sticky="w")

        steps = (
            ("01", "Escolha a obra"),
            ("02", "Configure a LLM"),
            ("03", "Acompanhe a execução"),
            ("04", "Revise os arquivos"),
        )
        for row, (number, label) in enumerate(steps, start=3):
            self._add_workflow_step(sidebar, row, number, label)

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.grid(row=9, column=0, padx=30, pady=28, sticky="sew")
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            footer,
            text="Tema claro",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["muted"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkSwitch(
            footer,
            text="",
            width=38,
            variable=self.theme_var,
            onvalue="light",
            offvalue="dark",
            command=self._toggle_theme,
            progress_color=COLORS["accent"],
        ).grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(
            footer,
            text="Projeto open source · PT-BR",
            font=ctk.CTkFont(size=10),
            text_color="#777B9A",
        ).grid(row=1, column=0, columnspan=2, pady=(18, 0), sticky="w")

    @staticmethod
    def _add_workflow_step(
        parent: ctk.CTkFrame,
        row: int,
        number: str,
        label: str,
    ) -> None:
        """Adiciona uma etapa compacta ao resumo lateral."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, padx=28, pady=7, sticky="ew")
        ctk.CTkLabel(
            frame,
            text=number,
            width=30,
            height=30,
            corner_radius=9,
            fg_color=COLORS["surface_strong"],
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=10, weight="bold"),
        ).grid(row=0, column=0)
        ctk.CTkLabel(
            frame,
            text=label,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=1, padx=(12, 0), sticky="w")

    def _build_workspace(self) -> None:
        """Monta os cartões de entrada, configuração e execução."""
        workspace = ctk.CTkScrollableFrame(
            self,
            corner_radius=0,
            fg_color=COLORS["bg"],
            scrollbar_button_color=COLORS["surface_strong"],
            scrollbar_button_hover_color=COLORS["blue"],
        )
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(workspace, fg_color="transparent")
        header.grid(row=0, column=0, padx=34, pady=(30, 20), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Tradução assistida por IA",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Do idioma original para português brasileiro, com QA e rastreabilidade.",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).grid(row=1, column=0, pady=(5, 0), sticky="w")
        ctk.CTkLabel(
            header,
            text="OPEN SOURCE",
            width=96,
            height=28,
            corner_radius=14,
            fg_color=COLORS["surface_strong"],
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=9, weight="bold"),
        ).grid(row=0, column=1, rowspan=2, padx=(20, 0), sticky="e")

        self._build_input_card(workspace)
        options = ctk.CTkFrame(workspace, fg_color="transparent")
        options.grid(row=2, column=0, padx=34, pady=(0, 18), sticky="ew")
        options.grid_columnconfigure((0, 1), weight=1, uniform="options")
        self._build_model_card(options)
        self._build_delivery_card(options)
        self._build_execution_card(workspace)

    def _build_input_card(self, parent: ctk.CTkFrame) -> None:
        """Cria a seleção da obra e do idioma de origem."""
        card = self._new_card(parent)
        card.grid(row=1, column=0, padx=34, pady=(0, 18), sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        self._add_card_title(
            card,
            "Arquivo de origem",
            "PDF, Markdown ou TXT. O formato define automaticamente o fluxo de entrada.",
        )

        input_frame = ctk.CTkFrame(card, fg_color="transparent")
        input_frame.grid(row=2, column=0, columnspan=2, padx=22, pady=(18, 8), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        self.input_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.input_var,
            height=42,
            corner_radius=10,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_deep"],
            placeholder_text="Selecione a obra que será traduzida",
        )
        self.input_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            input_frame,
            text="Escolher arquivo",
            width=132,
            height=42,
            corner_radius=10,
            command=self._select_input_file,
            fg_color=COLORS["blue"],
            hover_color=COLORS["accent"],
        ).grid(row=0, column=1, padx=(10, 0))

        language_frame = ctk.CTkFrame(card, fg_color="transparent")
        language_frame.grid(row=3, column=0, columnspan=2, padx=22, pady=(8, 22), sticky="ew")
        language_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            language_frame,
            text="Idioma de origem",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, padx=(2, 16), sticky="w")
        ctk.CTkOptionMenu(
            language_frame,
            variable=self.language_var,
            values=self.language_options,
            height=38,
            corner_radius=9,
            fg_color=COLORS["surface_strong"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["accent"],
        ).grid(row=0, column=1, sticky="ew")

    def _build_model_card(self, parent: ctk.CTkFrame) -> None:
        """Cria os campos de backend, modelo e credencial temporária."""
        card = self._new_card(parent)
        card.grid(row=0, column=0, padx=(0, 9), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        self._add_card_title(
            card, "Modelo de tradução", "Escolha execução local ou uma API externa."
        )

        self._add_field_label(card, 2, "Backend")
        self.backend_menu = ctk.CTkOptionMenu(
            card,
            variable=self.backend_var,
            values=list(BACKEND_LABELS),
            command=self._on_backend_change,
            height=38,
            fg_color=COLORS["surface_strong"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["accent"],
        )
        self.backend_menu.grid(row=3, column=0, padx=22, sticky="ew")

        self._add_field_label(card, 4, "Modelo")
        self.model_entry = ctk.CTkEntry(
            card,
            textvariable=self.model_var,
            height=38,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_deep"],
            placeholder_text="Modelo disponível no backend",
        )
        self.model_entry.grid(row=5, column=0, padx=22, sticky="ew")

        self.api_key_label = ctk.CTkLabel(
            card,
            text="Chave de API",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.api_key_label.grid(row=6, column=0, padx=24, pady=(14, 6), sticky="w")
        self.api_key_entry = ctk.CTkEntry(
            card,
            textvariable=self.api_key_var,
            show="•",
            height=38,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_deep"],
            placeholder_text="Usada somente nesta execução",
        )
        self.api_key_entry.grid(row=7, column=0, padx=22, pady=(0, 22), sticky="ew")

    def _build_delivery_card(self, parent: ctk.CTkFrame) -> None:
        """Cria as opções editoriais e de saída do trabalho."""
        card = self._new_card(parent)
        card.grid(row=0, column=1, padx=(9, 0), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        self._add_card_title(
            card, "Acabamento e saída", "Ative apenas as etapas necessárias para a obra."
        )

        switches = (
            ("QA e reparo seletivo", self.repair_var),
            ("Refino literário com LLM", self.refine_var),
            ("Gerar PDF ao finalizar", self.pdf_var),
            ("Retomar execução anterior", self.resume_var),
            ("Salvar diagnóstico detalhado", self.debug_var),
        )
        for row, (label, variable) in enumerate(switches, start=2):
            ctk.CTkSwitch(
                card,
                text=label,
                variable=variable,
                progress_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                font=ctk.CTkFont(size=12),
            ).grid(row=row, column=0, padx=22, pady=6, sticky="w")

        glossary_row = ctk.CTkFrame(card, fg_color="transparent")
        glossary_row.grid(row=7, column=0, padx=22, pady=(10, 4), sticky="ew")
        glossary_row.grid_columnconfigure(1, weight=1)
        ctk.CTkSwitch(
            glossary_row,
            text="Glossário",
            variable=self.use_glossary_var,
            command=self._refresh_glossary_fields,
            progress_color=COLORS["accent"],
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="w")
        self.glossary_button = ctk.CTkButton(
            glossary_row,
            text="Selecionar JSON",
            width=112,
            height=30,
            command=self._select_glossary_file,
            fg_color=COLORS["surface_strong"],
            hover_color=COLORS["blue"],
            state="disabled",
        )
        self.glossary_button.grid(row=0, column=1, sticky="e")
        self.glossary_label = ctk.CTkLabel(
            card,
            text="Nenhum glossário selecionado",
            text_color="#777B9A",
            font=ctk.CTkFont(size=10),
        )
        self.glossary_label.grid(row=8, column=0, padx=24, pady=(0, 20), sticky="w")

    def _build_execution_card(self, parent: ctk.CTkFrame) -> None:
        """Cria os controles de execução e o console de acompanhamento."""
        card = self._new_card(parent)
        card.grid(row=3, column=0, padx=34, pady=(0, 34), sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        self._add_card_title(
            card,
            "Execução",
            "A chave de API não aparece no comando nem é gravada em arquivo.",
        )

        settings = ctk.CTkFrame(card, fg_color="transparent")
        settings.grid(row=2, column=0, columnspan=2, padx=22, pady=(16, 12), sticky="ew")
        settings.grid_columnconfigure(0, weight=1)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            settings,
            text="Preparação do PDF",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        self.desquebrar_menu = ctk.CTkOptionMenu(
            settings,
            variable=self.desquebrar_var,
            values=list(DESQUEBRAR_LABELS),
            height=36,
            fg_color=COLORS["surface_strong"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["accent"],
        )
        self.desquebrar_menu.grid(row=1, column=0, padx=(0, 10), pady=(6, 0), sticky="ew")

        ctk.CTkLabel(
            settings,
            text="Timeout por chamada (segundos)",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=0, column=1, padx=(10, 0), sticky="w")
        ctk.CTkEntry(
            settings,
            textvariable=self.timeout_var,
            height=36,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_deep"],
        ).grid(row=1, column=1, padx=(10, 0), pady=(6, 0), sticky="ew")

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=3, column=0, columnspan=2, padx=22, pady=(4, 14), sticky="ew")
        actions.grid_columnconfigure(1, weight=1)
        self.start_button = ctk.CTkButton(
            actions,
            text="Iniciar tradução",
            width=160,
            height=44,
            corner_radius=11,
            command=self._start_translation,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["accent_text"],
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.start_button.grid(row=0, column=0)
        self.cancel_button = ctk.CTkButton(
            actions,
            text="Cancelar",
            width=96,
            height=44,
            command=self._cancel_translation,
            fg_color="transparent",
            border_width=1,
            border_color=COLORS["danger"],
            text_color=COLORS["danger"],
            hover_color=COLORS["surface_strong"],
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=(10, 0), sticky="w")
        self.output_button = ctk.CTkButton(
            actions,
            text="Abrir pasta de saída",
            width=146,
            height=36,
            command=self._open_output_directory,
            fg_color=COLORS["surface_strong"],
            hover_color=COLORS["blue"],
        )
        self.output_button.grid(row=0, column=2, sticky="e")

        status_row = ctk.CTkFrame(card, fg_color="transparent")
        status_row.grid(row=4, column=0, columnspan=2, padx=22, sticky="ew")
        status_row.grid_columnconfigure(1, weight=1)
        self.status_dot = ctk.CTkLabel(
            status_row,
            text="●",
            width=14,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        )
        self.status_dot.grid(row=0, column=0)
        self.status_label = ctk.CTkLabel(
            status_row,
            text="Pronto para configurar",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        )
        self.status_label.grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.progress = ctk.CTkProgressBar(
            status_row,
            width=180,
            height=8,
            progress_color=COLORS["accent"],
            fg_color=COLORS["surface_strong"],
            mode="indeterminate",
        )
        self.progress.grid(row=0, column=2, sticky="e")
        self.progress.set(0)

        self.log_text = ctk.CTkTextbox(
            card,
            height=190,
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["bg_deep"],
            text_color=COLORS["log_text"],
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
        )
        self.log_text.grid(row=5, column=0, columnspan=2, padx=22, pady=(10, 22), sticky="ew")
        self.log_text.insert("end", "O andamento do pipeline aparecerá aqui.\n")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _new_card(parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Cria um cartão com a superfície padrão da identidade visual."""
        return ctk.CTkFrame(
            parent,
            corner_radius=16,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["surface"],
        )

    @staticmethod
    def _add_card_title(card: ctk.CTkFrame, title: str, subtitle: str) -> None:
        """Adiciona título e descrição ao topo de um cartão."""
        ctk.CTkLabel(
            card,
            text=title,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=22, pady=(20, 0), sticky="w")
        ctk.CTkLabel(
            card,
            text=subtitle,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, columnspan=2, padx=22, pady=(4, 0), sticky="w")

    @staticmethod
    def _add_field_label(card: ctk.CTkFrame, row: int, text: str) -> None:
        """Adiciona o rótulo discreto usado nos campos de configuração."""
        ctk.CTkLabel(
            card,
            text=text,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=row, column=0, padx=24, pady=(14, 6), sticky="w")

    def _select_input_file(self) -> None:
        """Abre o seletor de arquivos e atualiza controles dependentes do formato."""
        selected = filedialog.askopenfilename(
            title="Escolha a obra",
            filetypes=(
                ("Entradas compatíveis", "*.pdf *.md *.txt"),
                ("PDF", "*.pdf"),
                ("Markdown", "*.md"),
                ("Texto", "*.txt"),
                ("Todos os arquivos", "*.*"),
            ),
        )
        if not selected:
            return
        self.input_var.set(selected)
        is_pdf = Path(selected).suffix.casefold() == ".pdf"
        self.desquebrar_menu.configure(state="normal" if is_pdf else "disabled")

    def _select_glossary_file(self) -> None:
        """Seleciona um glossário JSON e exibe apenas seu nome na tela."""
        selected = filedialog.askopenfilename(
            title="Escolha o glossário",
            filetypes=(("Glossário JSON", "*.json"), ("Todos os arquivos", "*.*")),
        )
        if not selected:
            return
        self.glossary_var.set(selected)
        self.glossary_label.configure(text=Path(selected).name, text_color=COLORS["muted"])

    def _refresh_glossary_fields(self) -> None:
        """Habilita a seleção de glossário somente quando a opção estiver ativa."""
        state = "normal" if self.use_glossary_var.get() else "disabled"
        self.glossary_button.configure(state=state)
        if not self.use_glossary_var.get():
            self.glossary_label.configure(text_color="#777B9A")

    def _on_backend_change(self, selected_label: str) -> None:
        """Preserva o modelo anterior de cada backend e atualiza a credencial exigida."""
        current_backend = getattr(self, "current_backend", self.config_data.translate_backend)
        self.backend_models[current_backend] = self.model_var.get().strip()
        new_backend = BACKEND_LABELS[selected_label]
        self.model_var.set(self.backend_models[new_backend])
        self.current_backend = new_backend
        self._refresh_backend_fields()

    def _refresh_backend_fields(self) -> None:
        """Adapta o campo de chave para execução local, Gemini ou OpenAI."""
        backend = BACKEND_LABELS[self.backend_var.get()]
        self.current_backend = backend
        environment_name = required_api_key_environment(backend)
        if environment_name:
            self.api_key_label.configure(text=f"Chave de API · {environment_name}")
            self.api_key_entry.configure(state="normal")
        else:
            self.api_key_label.configure(text="Chave de API · não necessária no Ollama")
            self.api_key_entry.configure(state="disabled")
            self.api_key_var.set("")

    def _build_job_from_form(self) -> TranslationJob:
        """Converte os valores da tela em uma descrição imutável da execução."""
        try:
            timeout = int(self.timeout_var.get().strip())
        except ValueError:
            timeout = 0
        glossary_path = None
        if self.use_glossary_var.get() and self.glossary_var.get().strip():
            glossary_path = Path(self.glossary_var.get().strip())
        return TranslationJob(
            input_path=Path(self.input_var.get().strip()),
            source_language=self.language_codes[self.language_var.get()],
            backend=BACKEND_LABELS[self.backend_var.get()],
            model=self.model_var.get(),
            request_timeout=timeout,
            refine=self.refine_var.get(),
            repair=self.repair_var.get(),
            export_pdf=self.pdf_var.get(),
            resume=self.resume_var.get(),
            debug=self.debug_var.get(),
            glossary_path=glossary_path,
            desquebrar_mode=DESQUEBRAR_LABELS[self.desquebrar_var.get()],
        )

    def _start_translation(self) -> None:
        """Valida o formulário e inicia o processo da CLI sem bloquear a janela."""
        job = self._build_job_from_form()
        errors = validate_translation_job(job)
        environment_name = required_api_key_environment(job.backend)
        api_key = self.api_key_var.get().strip()
        if environment_name and not (api_key or os.getenv(environment_name)):
            errors.append(f"Informe {environment_name} para usar esse backend.")
        if self.use_glossary_var.get() and not job.glossary_path:
            errors.append("Selecione o arquivo JSON do glossário ou desative essa opção.")
        if errors:
            messagebox.showerror(
                "Revise a configuração", "\n".join(f"• {error}" for error in errors)
            )
            return

        command = build_cli_command(job)
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        if environment_name and api_key:
            environment[environment_name] = api_key

        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            messagebox.showerror("Não foi possível iniciar", str(exc))
            return

        self.active_job = job
        self.cancel_requested = False
        self._set_running_state(True)
        self._replace_log("Execução iniciada. Aguardando a primeira etapa do pipeline…\n")
        self.worker = threading.Thread(target=self._read_process_output, daemon=True)
        self.worker.start()

    def _read_process_output(self) -> None:
        """Transfere a saída do subprocesso para uma fila segura para a thread da GUI."""
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.message_queue.put(("log", line))
        return_code = process.wait()
        self.message_queue.put(("finished", return_code))

    def _drain_message_queue(self) -> None:
        """Consome eventos do worker e agenda a próxima verificação da fila."""
        try:
            while True:
                event, payload = self.message_queue.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "finished":
                    self._finish_translation(int(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain_message_queue)

    def _finish_translation(self, return_code: int) -> None:
        """Restaura a janela e comunica o resultado final da CLI."""
        self._set_running_state(False)
        if self.cancel_requested:
            self._set_status("Execução cancelada", COLORS["danger"])
            self._append_log("\nExecução cancelada pelo usuário.\n")
        elif return_code == 0:
            self._set_status("Tradução concluída", COLORS["success"])
            self._append_log("\nPipeline concluído com sucesso.\n")
            existing_outputs = self._existing_outputs()
            if existing_outputs:
                self._append_log("Arquivos principais:\n")
                for output in existing_outputs:
                    self._append_log(f"  • {output}\n")
        else:
            self._set_status(f"Falha na execução · código {return_code}", COLORS["danger"])
            self._append_log("\nO pipeline encerrou com erro. Consulte as últimas linhas acima.\n")
        self.process = None
        self.worker = None

    def _existing_outputs(self) -> tuple[Path, ...]:
        """Filtra as saídas previstas que realmente foram gravadas no disco."""
        if self.active_job is None:
            return ()
        output_dir = self.config_data.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        return tuple(
            path for path in expected_output_paths(self.active_job, output_dir) if path.exists()
        )

    def _cancel_translation(self) -> None:
        """Solicita o encerramento do processo em execução."""
        if self.process is None or self.process.poll() is not None:
            return
        self.cancel_requested = True
        self._set_status("Cancelando…", COLORS["danger"])
        self.cancel_button.configure(state="disabled")
        self.process.terminate()

    def _set_running_state(self, running: bool) -> None:
        """Alterna botões, progresso e mensagem conforme o estado da execução."""
        self.start_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start()
            self._set_status("Pipeline em execução", COLORS["accent"])
        else:
            self.progress.stop()
            self.progress.set(0)

    def _set_status(self, text: str, color: ColorValue) -> None:
        """Atualiza de forma consistente o texto e o indicador de status."""
        self.status_label.configure(text=text)
        self.status_dot.configure(text_color=color)

    def _replace_log(self, text: str) -> None:
        """Substitui o conteúdo do console por uma nova mensagem."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", text)
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        """Acrescenta texto ao console e mantém a visualização no fim."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_output_directory(self) -> None:
        """Abre o diretório de saída usando o gerenciador de arquivos do sistema."""
        output_dir = self.config_data.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])

    def _toggle_theme(self) -> None:
        """Alterna o modo de aparência dos widgets da aplicação."""
        ctk.set_appearance_mode(self.theme_var.get())

    def _on_close(self) -> None:
        """Confirma o fechamento quando ainda existe uma tradução em andamento."""
        if self.process is not None and self.process.poll() is None:
            should_close = messagebox.askyesno(
                "Tradução em andamento",
                "Encerrar a tradução e fechar a interface?",
            )
            if not should_close:
                return
            self.process.terminate()
        self.destroy()
