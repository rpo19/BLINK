CREATE TABLE IF NOT EXISTS entities (
    id INT NOT NULL,
    indexer INT NOT NULL,
    wikipedia_id INT NOT NULL,
    wikidata_qid INT,
    title varchar(100) NOT NULL,
    descr TEXT,
    type_ varchar(20),
    embedding TEXT,
    PRIMARY KEY (id, indexer)
);

CREATE INDEX IF NOT EXISTS entities_wikipedia_id ON public.entities USING btree (wikipedia_id);
CREATE INDEX IF NOT EXISTS entities_wikidata_qid ON public.entities USING btree (wikidata_qid);
