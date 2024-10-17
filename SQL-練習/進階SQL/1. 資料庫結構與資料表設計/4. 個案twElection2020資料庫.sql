CREATE DATABASE tw_election_2020 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 設定主鍵
ALTER TABLE tw_election_2020.admin_regions ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE tw_election_2020.candidates ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE tw_election_2020.parties ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE tw_election_2020.presidential ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE tw_election_2020.legislative_regional ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE tw_election_2020.legislative_at_large ADD CONSTRAINT PRIMARY KEY (id);


-- 設定外鍵
ALTER TABLE tw_election_2020.candidates
ADD CONSTRAINT fk_candidates_parties FOREIGN KEY (party_id) REFERENCES parties (id);

ALTER TABLE tw_election_2020.presidential
ADD CONSTRAINT fk_presidientail_admin_regions FOREIGN KEY (admin_region_id) REFERENCES admin_regions (id),
ADD CONSTRAINT fk_presidientail_candidates FOREIGN KEY (candidate_id) REFERENCES candidates (id);

ALTER TABLE tw_election_2020.legislative_regional
ADD CONSTRAINT fk_legislative_regional_admin_regions FOREIGN KEY (admin_region_id) REFERENCES admin_regions (id),
ADD CONSTRAINT fk_legislative_regional_candidates FOREIGN KEY (candidate_id) REFERENCES candidates (id);

ALTER TABLE tw_election_2020.legislative_at_large
ADD CONSTRAINT fk_legislative_at_large_admin_regions FOREIGN KEY (admin_region_id) REFERENCES admin_regions (id),
ADD CONSTRAINT fk_legislative_at_large_parties FOREIGN KEY (party_id) REFERENCES parties (id);






