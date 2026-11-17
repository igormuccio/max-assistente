CREATE TABLE clientes (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(150) NOT NULL,
    email TEXT UNIQUE,
    criado_em TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT clientes_nome_check CHECK (TRIM(BOTH FROM nome) <> ''),
    CONSTRAINT clientes_email_check CHECK (email IS NULL OR TRIM(BOTH FROM email) <> '')
);

CREATE TABLE conversas (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    iniciada_em TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE mensagens (
    id SERIAL PRIMARY KEY,
    conversa_id INTEGER NOT NULL REFERENCES conversas(id),
    remetente VARCHAR(10) NOT NULL,
    conteudo TEXT NOT NULL,
    enviada_em TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT mensagens_conteudo_check CHECK (TRIM(BOTH FROM conteudo) <> ''),
    CONSTRAINT mensagens_remetente_check CHECK (remetente IN ('cliente', 'max'))
);