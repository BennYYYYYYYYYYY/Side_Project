CREATE DATABASE covid19 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 更新資料
UPDATE covid19.locations
SET iso2 = 'NA'
WHERE country_name = 'Namibia';

UPDATE covid19.locations
SET iso2 = NULL,
    iso3 = NULL
WHERE iso2 = '' AND
      iso3 = '';

UPDATE covid19.locations
SET province_name = NULL
WHERE province_name = '';


-- 設定主鍵
ALTER TABLE covid19.accumulative_cases ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE covid19.calendars ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE covid19.locations ADD CONSTRAINT PRIMARY KEY (id);


-- 設定外鍵
ALTER TABLE covid19.accumulative_cases
ADD CONSTRAINT fk_accumulative_cases_calendars FOREIGN KEY (calendar_id) REFERENCES calendars(id),
ADD CONSTRAINT fk_accumulative_cases_locations FOREIGN KEY (location_id) REFERENCES locations (id);

   






