# Conception — Chat multi-tour (sessions, agent LangChain, streaming, UI Gradio)

Date : 2026-11-08 · Statut : brouillon, à valider

## Intention

Le POC doit permettre une **conversation** sur les événements culturels, pas seulement
une question isolée. L'utilisateur pose un cadre (ville, période), puis affine sans le
répéter ; l'assistant choisit lui-même ses outils (recherche filtrée dans l'index régional),
élargit la zone quand il ne trouve rien et le dit.

Conversation de référence (fournie par William) :

```
User   : Bonjour, j'aimerais voir les évènements sur Aix-en-Provence dans les 15 prochains jours.
System : Il y a 8 événements sur Aix-en-Provence à venir avec pour sujet des entretiens de
         recrutement, de la formation, des ateliers. Que recherchez-vous exactement ?
User   : Je recherche un atelier d'immersion professionnelle.
System : Pas d'atelier d'immersion professionnelle sur Aix-en-Provence les 15 prochains jours ;
         en revanche il y a un atelier prévu le mardi 3 novembre à 09h00 à l'agence
         Carré Gabriel, Marseille. [...] souhaitez-vous continuer votre recherche ?
```

Ce que cela implique :
- au tour 2, la ville et la période du tour 1 sont **réutilisées** sans être redites ;
- l'assistant résume les thèmes trouvés et pose une question d'affinage ;
- en l'absence de résultat, il **élargit** aux communes voisines et l'annonce clairement.

Le cadrage OpenClassrooms n'exige que l'API FastAPI. Une petite UI **Gradio** est ajoutée
comme outil de dev, pour valider l'interaction back ↔ front comme le ferait un vrai client.

## Objectifs

1. Un chat = une **session**, propriété d'un utilisateur authentifié (JWT existant).
2. Mémoire multi-tour persistante (survit au redémarrage de l'API).
3. Agent LangChain à outils : le LLM décide quel outil appeler et avec quels arguments.
4. Réponses **streamées** (SSE) : texte au fil de l'eau, appels d'outils, sources citées.
5. UI Gradio de dev qui ne passe **que** par l'API HTTP.

## Non-objectifs

- **Ajout à l'agenda** de l'utilisateur (liste en base, export `.ics`) : hors POC ; ajout
  éventuel avant la soutenance, dans un lot séparé. L'assistant peut ne pas le proposer.
- Front de production (Next.js ou autre).
- Configuration des outils par session : tous les outils sont disponibles pour toutes les
  sessions.
- Partage de sessions entre utilisateurs, rôles, administration.

## Anti-objectifs (échec même si « ça marche »)

- L'UI Gradio importe l'agent ou la base directement (elle ne validerait plus rien).
- Un utilisateur peut lire, écrire ou supprimer la session d'un autre.
- Une table de messages maison qui duplique ce que LangGraph sait déjà persister.
- Des tests unitaires qui appellent le réseau (LLM, embeddings, Open Agenda).
- Un modèle à réflexion (contrainte AGENTS.md : profils `local`/`ci` sans réflexion).

## Contraintes

- Python 3.11, FastAPI 0.162 (SSE natif : `fastapi.sse.EventSourceResponse`), SQLModel,
  Pydantic 2, `langchain` 1.4, `langgraph` 1.2 (déjà installés).
- **Maximiser l'usage des briques LangChain/LangGraph** (demande de William).
- LLM via l'adaptateur existant `LangOpenAI` / `build_client()` (`langchain-openai`,
  profils `LLM_PROFILE`) ; embeddings `CachedEmbeddings` existants.
- Le modèle de chat doit supporter le *tool calling* dans les deux profils (vLLM Qwen,
  OpenRouter Mistral Small) ; à vérifier en test d'intégration.
- Tests unitaires sans réseau (LLM simulé, par ex. `GenericFakeChatModel` /
  fake model à tool calls, `httpx.MockTransport` pour Open Agenda).
- Workflow semver-batch (issue typée, sous-branche, PR de lot), textes en français.
- Image Docker : n'embarque pas Gradio.

## Décisions (prises avec William)

| Sujet | Décision |
|---|---|
| Unité de conversation | une session `session_id` rattachée à `user_id` |
| Outils | recherche filtrée dans un index régional complet ; mêmes outils pour toutes les sessions |
| Index | tout l'agenda de la région, synchronisé par lot, valide 24 h ; filtrage au retrieval |
| Agenda | hors POC |
| Framework | LangChain au maximum |
| Client de dev | UI Gradio, via l'API HTTP uniquement |
| Réponse | streaming dès la première version |

## Conception

### 0. Index vectoriel régional (synchronisation par lot, 24 h)

Changement de cap décidé par William : on n'indexe plus « à la volée » les quelques
résultats d'une requête Open Agenda. **Tout l'agenda de la région** (agenda `21769447`,
événements des 365 derniers jours et à venir) est collecté, découpé et vectorisé hors
ligne, par lot ; le filtrage par mots-clés et par localité se fait **au retrieval**, sur
le vector store, et non plus dans l'appel à l'API.

- **Collecte** (#5) : pagination complète de l'API v2 (`size=300`, curseur `after`),
  `detailed=1`, `timings[gte]` = aujourd'hui − 365 jours. ≈ 4 300 événements, ≈ 15 pages.
- **Documents** (#6) : `event_to_documents` existant, métadonnées enrichies pour le
  filtrage : commune normalisée (`communes.json`), `lat`/`lng`, dates de début et de fin
  (ISO, comparables), mots-clés / catégories, `uid`, `url`.
- **Index** (#7) : Faiss LangChain sauvegardé dans `data/index/` avec un manifeste
  (modèle d'embeddings, dimension, nombre d'événements et de passages, date de
  synchronisation, empreinte des données). Refus de charger un index construit avec un
  autre modèle d'embeddings. Les embeddings passent par `CachedEmbeddings` (30 jours) :
  une resynchronisation quotidienne ne paie que les passages nouveaux ou modifiés.
- **Fraîcheur : 24 h.** L'index est valide 24 h après sa date de synchronisation
  (`INDEX_TTL_HOURS`, défaut 24), ce qui garantit qu'il ne contient que des événements de
  moins d'un an.
- **Vérification à la création d'une session** : le serveur regarde la date de l'index.
  - index présent et frais → utilisé tel quel ;
  - index présent mais expiré → **la session est créée immédiatement sur l'index
    actuel**, et une resynchronisation est lancée en tâche de fond *(mon choix : une
    collecte + indexation complète prend plusieurs minutes, on ne bloque pas
    l'utilisateur)* ; le nouvel index remplace l'ancien de façon atomique quand il est
    prêt (même mécanique que `/rebuild`, #25) ; une seule synchronisation à la fois ;
  - aucun index → la synchronisation est lancée et la création de session répond `503`
    avec un message explicite tant qu'aucun index n'existe *(mon choix ; l'image Docker
    embarque un index, #11, donc ce cas ne concerne que le premier lancement en dev)*.
- Commande hors ligne : `scripts/build_index.py` (collecte → préparation → index) ;
  `POST /rebuild` (#25) réutilise la même fonction.
- Le même index sert `POST /ask` (#9, #10) et les outils du chat.

### 1. Persistance

**Table `chat_sessions` (SQLModel, `database/models.py`)** — métadonnées et propriété :

| Champ | Type | Note |
|---|---|---|
| `session_id` | `uuid.UUID` | clé primaire, `default_factory=uuid4` |
| `user_id` | `int` | clé étrangère `users.id`, indexée, non nulle |
| `title` | `str \| None` | ≤ 200 car. ; par défaut, début du premier message |
| `created_at` | `datetime` (UTC) | |
| `updated_at` | `datetime` (UTC) | mis à jour à chaque message |

**Historique des messages : checkpointer LangGraph**, `thread_id = str(session_id)`.
*(Mon choix)* `AsyncSqliteSaver` (`langgraph-checkpoint-sqlite`, nouvelle dépendance) pour
survivre au redémarrage. Fichier dédié configurable `CHECKPOINT_DB`
(défaut `./data/checkpoints.db`, donc dans le volume `/app/data` de l'image), distinct de
`database.db` : le checkpointer gère son propre schéma. Ouvert/fermé dans le `lifespan`.
En test : `InMemorySaver`.

Suppression d'une session : supprimer la ligne **et** le fil du checkpointer
(`adelete_thread`). Suppression d'un utilisateur : ses sessions et leurs fils aussi
*(mon choix : fait par le service, pas par `ON DELETE CASCADE`, pour nettoyer le
checkpointer)*.

### 2. Schémas Pydantic de l'API (`chat/schemas.py`)

- `ChatSessionCreate` : `title: str | None`.
- `ChatSessionRead` : `session_id`, `title`, `created_at`, `updated_at`.
- `ChatMessageIn` : `content: str` (1 à 2 000 caractères, non vide après `strip`).
- `ChatMessage` (historique) : `role: Literal["user", "assistant", "tool"]`, `content: str`,
  `tool_name: str | None`, `sources: list[EventSource]` (vide sauf réponses assistant).
- `ChatSessionDetail` : `ChatSessionRead` + `messages: list[ChatMessage]`, reconstruit
  depuis l'état du checkpointer (`aget_state`).
- `EventSource` : `uid`, `title`, `date_range`, `location`, `url`, `distance_km: float | None`.
- Événements SSE (un modèle par type, union discriminée sur `type`) :
  - `token` : `{"type": "token", "content": "..."}` — fragment de texte de l'assistant ;
  - `tool_start` : `{"type": "tool_start", "tool": "search_events", "args": {...}}` ;
  - `tool_end` : `{"type": "tool_end", "tool": "...", "summary": "8 événements"}` ;
  - `sources` : `{"type": "sources", "events": [EventSource, ...]}` — événements cités ;
  - `done` : `{"type": "done"}` ;
  - `error` : `{"type": "error", "detail": "..."}` (l'erreur arrive dans le flux, le
    statut HTTP étant déjà 200).

### 3. Agent (`chat/agent.py`)

- `langchain.agents.create_agent(model, tools, system_prompt=..., checkpointer=...,
  state_schema=...)`, construit **une fois** au démarrage, réutilisé par toutes les
  sessions (l'isolation vient du `thread_id`).
- Modèle : `build_client()` existant (profil `LLM_PROFILE`, réflexion désactivée).
- Prompt système en français : rôle (assistant Puls-Events), **date du jour injectée à
  chaque tour** (nécessaire pour « les 15 prochains jours »), règles : résumer les thèmes
  et poser une question d'affinage quand les résultats sont nombreux ; annoncer
  explicitement un élargissement géographique ; ne citer que des événements renvoyés
  par les outils ; ne pas proposer d'ajout à l'agenda.
- Pas d'état d'agent spécifique : les outils interrogent l'**index régional partagé**
  (section 0). Les filtres du tour précédent (ville, période) sont retrouvés par le LLM
  dans l'historique des messages, y compris les arguments de ses propres appels d'outils.
- Fenêtre de contexte : *(mon choix)* middleware de résumé/troncature de LangChain si
  disponible, sinon limite aux N derniers messages ; à régler par le builder.

### 4. Outils (`chat/tools.py`, `@tool` LangChain avec `args_schema` Pydantic)

Les deux outils interrogent l'index régional (section 0), **jamais l'API Open Agenda**.

1. **`search_events`** — `query: str` (thème, type d'événement, mots-clés libres),
   `cities: list[str]`, `since: date | None`, `until: date | None`, `k: int = 10`.
   Recherche par similarité dans le vector store avec **filtre de métadonnées**
   (commune, chevauchement de la période avec les dates de l'événement). Si les communes
   demandées ne donnent rien, élargissement géographique **sur les métadonnées** : rayons
   successifs `FALLBACK_RADII_KM` autour du centre des communes (`lat`/`lng` des
   documents, `haversine_km`), résultats annotés de `distance_km`, classés par pertinence (rayon le plus petit qui donne un résultat).
   Le résumé renvoyé au LLM indique explicitement l'élargissement (« rien à Aix, 3 à
   moins de 20 km »).
   **Couverture géographique** *(mon choix)* : l'agenda `21769447` couvre la métropole
   Aix-Marseille-Provence (`communes.json`). Chaque nom de `cities` est normalisé via le
   référentiel (alias « Aix » → « Aix-en-Provence ») ; un nom hors référentiel est
   signalé au LLM. Le prompt système indique que l'assistant ne couvre que la métropole.
2. **`list_event_themes`** *(mon choix)* — `cities`, `since`, `until` : compte les
   événements correspondant aux filtres et renvoie les thèmes dominants (mots-clés /
   catégories des métadonnées). Sert au premier tour de l'exemple (« 8 événements, sur
   le recrutement, la formation, les ateliers ; que cherchez-vous ? ») sans similarité
   sémantique, la requête n'ayant pas encore de thème.

*Note builder* : avec Faiss, le filtre de métadonnées de LangChain s'applique **après**
la recherche des `fetch_k` voisins. L'index ne compte que quelques milliers d'événements
(≈ 4 300 sur 365 jours, mesuré le 2026-11-08) : soit `fetch_k` couvre tout l'index, soit
on pré-sélectionne les identifiants par filtre puis on calcule la similarité sur ce
sous-ensemble. À trancher par le builder, l'exigence est que le filtre soit **exhaustif**
(aucun événement d'Aix manqué parce qu'il était hors des `fetch_k` premiers).

**Sources** *(mon choix)* : `sources` = tous les événements **renvoyés par les outils**
pendant le tour (dédoublonnés par `uid`), pas seulement ceux que le texte cite — pas de
sortie structurée imposée au LLM, streaming simple. Chaque outil renvoie
`content_and_artifact` : le résumé texte va au LLM, la liste d'`EventSource` est
l'artefact du `ToolMessage`, donc persistée par le checkpointer et relue pour
reconstruire les sources des anciens tours dans `ChatSessionDetail`.

### 5. Routes (`chat/router.py`, préfixe `/chat`, tag `chat`, toutes `ActiveUser`)

| Méthode | Chemin | Effet |
|---|---|---|
| `POST` | `/chat/sessions` | crée une session → `201 ChatSessionRead` |
| `GET` | `/chat/sessions` | sessions de l'utilisateur, `updated_at` décroissant (`offset`/`limit`) |
| `GET` | `/chat/sessions/{session_id}` | `ChatSessionDetail` (historique) |
| `POST` | `/chat/sessions/{session_id}/messages` | envoie un message → flux SSE |
| `DELETE` | `/chat/sessions/{session_id}` | `204`, supprime session et fil |

Session inexistante **ou appartenant à un autre utilisateur** : `404` (ne pas révéler
l'existence). Un seul message en cours par session *(mon choix : `409` si un flux est déjà
actif, verrou en mémoire du processus — suffisant pour le POC mono-processus)*.

Streaming : `agent.astream(..., stream_mode=["messages", "updates"])` traduit en
événements SSE de la section 2. Déconnexion du client : arrêter proprement la génération.

### 6. UI Gradio (`scripts/chat_ui.py`)

- Dépendance `gradio` dans le groupe `dev` (hors image Docker et hors `requirements.txt`
  de prod — à confirmer selon l'export actuel).
- Client `httpx` vers `API_URL` (défaut `http://localhost:8000`) ; port Gradio 7860.
- Connexion (défaut compte `demo`, cf. README) → `/auth/token`, jeton gardé en état Gradio.
- Liste des sessions (reprise + chargement de l'historique), bouton « Nouvelle conversation ».
- `gr.ChatInterface`/`gr.Chatbot` : lecture du flux SSE, affichage progressif des `token`,
  puis des sources (titre, date, lieu, lien).
- Panneau repliable « Outils appelés » (`tool_start`/`tool_end`) pour le débogage.

### 7. Tests

- Unitaires (sans réseau) : schémas ; isolation par utilisateur (404) ; CRUD sessions ;
  flux SSE complet avec un modèle factice qui émet un appel `search_events` puis une
  réponse ; outils avec `httpx.MockTransport` et embeddings factices ; multi-tour
  (au 2ᵉ message, l'agent réutilise ville et période du 1ᵉʳ) ; suppression de fil.
- Intégration (`@pytest.mark.integration`) : le scénario Aix-en-Provence de bout en bout
  avec le vrai LLM du profil courant. *(Mon choix)* Critères structurels seulement, les
  données Open Agenda changeant : au tour 1, `search_events` est appelé avec
  `cities=["Aix-en-Provence"]` et une période ≈ 15 jours à partir d'aujourd'hui ; au
  tour 2, l'agent n'a pas redemandé la ville et a appelé un outil ; le flux se termine par
  `done` sans `error`. Lancé à la main, **hors gate de release** (la qualité des réponses
  relève de l'évaluation #14).

## Laissé au builder

Noms internes, découpage fin des modules, format exact des résumés d'outils, stratégie de
fenêtre de contexte, génération du titre, mapping précis des chunks `astream` vers SSE.

## Livraison

Décidé avec William : le chat **coexiste** avec `POST /ask`.

- `/ask` (#10) et la chaîne RAG (#9) restent le livrable du cahier des charges et la
  cible de l'évaluation (#14) ; leur périmètre ne change pas.
- Le chat fait l'objet d'une **nouvelle issue `feat(chat)`** (lot minor), qui référence
  cette conception.
- L'index régional (#5, #6, #7) est un prérequis du chat et de `/ask`.
- Lot `feat/0.2.0`.
