CREATE TABLE jobs(
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  error TEXT,
  acceleration_report TEXT,
  updated_at TEXT NOT NULL DEFAULT ''
);
INSERT INTO jobs(job_id,status,acceleration_report) VALUES
('d0000','queued','{"state":"pending"}'),
('d0001','queued',NULL),
('d0002','queued','{"state":"pending"}'),
('d0003','queued','{"state":"determined","x":1}'),
('d0004','queued','{"state":"pending"}'),
('d0005','queued',NULL),
('d0006','queued','{"state":"pending"}'),
('d0007','queued','{"state":"determined","x":1}'),
('d0008','queued','{"state":"pending"}'),
('d0009','queued',NULL),
('d0010','queued','{"state":"pending"}'),
('d0011','queued','{"state":"determined","x":1}'),
('d0012','queued','{"state":"pending"}'),
('d0013','queued',NULL),
('d0014','queued','{"state":"pending"}'),
('d0015','queued','{"state":"determined","x":1}'),
('d0016','queued','{"state":"pending"}'),
('d0017','queued',NULL),
('d0018','queued','{"state":"pending"}'),
('d0019','queued','{"state":"determined","x":1}'),
('d0020','queued','{"state":"pending"}'),
('d0021','queued',NULL),
('d0022','queued','{"state":"pending"}'),
('d0023','queued','{"state":"determined","x":1}'),
('d0024','queued','{"state":"pending"}'),
('d0025','queued',NULL),
('d0026','queued','{"state":"pending"}'),
('d0027','queued','{"state":"determined","x":1}'),
('d0028','queued','{"state":"pending"}'),
('d0029','queued',NULL),
('d0030','queued','{"state":"pending"}'),
('d0031','queued','{"state":"determined","x":1}'),
('d0032','queued','{"state":"pending"}'),
('d0033','queued',NULL),
('d0034','queued','{"state":"pending"}'),
('d0035','queued','{"state":"determined","x":1}'),
('d0036','queued','{"state":"pending"}'),
('d0037','queued',NULL),
('d0038','queued','{"state":"pending"}'),
('d0039','queued','{"state":"determined","x":1}'),
('d0040','queued','{"state":"pending"}'),
('d0041','queued',NULL),
('d0042','queued','{"state":"pending"}'),
('d0043','queued','{"state":"determined","x":1}'),
('d0044','queued','{"state":"pending"}'),
('d0045','queued',NULL),
('d0046','queued','{"state":"pending"}'),
('d0047','queued','{"state":"determined","x":1}'),
('d0048','queued','{"state":"pending"}'),
('d0049','queued',NULL),
('d0050','queued','{"state":"pending"}'),
('d0051','queued','{"state":"determined","x":1}'),
('d0052','queued','{"state":"pending"}'),
('d0053','queued',NULL),
('d0054','queued','{"state":"pending"}'),
('d0055','queued','{"state":"determined","x":1}'),
('d0056','queued','{"state":"pending"}'),
('d0057','queued',NULL),
('d0058','queued','{"state":"pending"}'),
('d0059','queued','{"state":"determined","x":1}'),
('d0060','queued','{"state":"pending"}'),
('d0061','queued',NULL),
('d0062','queued','{"state":"pending"}'),
('d0063','queued','{"state":"determined","x":1}'),
('d0064','queued','{"state":"pending"}'),
('d0065','queued',NULL),
('d0066','queued','{"state":"pending"}'),
('d0067','queued','{"state":"determined","x":1}'),
('d0068','queued','{"state":"pending"}'),
('d0069','queued',NULL),
('d0070','queued','{"state":"pending"}'),
('d0071','queued','{"state":"determined","x":1}'),
('d0072','queued','{"state":"pending"}'),
('d0073','queued',NULL),
('d0074','queued','{"state":"pending"}'),
('d0075','queued','{"state":"determined","x":1}'),
('d0076','queued','{"state":"pending"}'),
('d0077','queued',NULL),
('d0078','queued','{"state":"pending"}'),
('d0079','queued','{"state":"determined","x":1}'),
('d0080','queued','{"state":"pending"}'),
('d0081','queued',NULL),
('d0082','queued','{"state":"pending"}'),
('d0083','queued','{"state":"determined","x":1}'),
('d0084','queued','{"state":"pending"}'),
('d0085','queued',NULL),
('d0086','queued','{"state":"pending"}'),
('d0087','queued','{"state":"determined","x":1}'),
('d0088','queued','{"state":"pending"}'),
('d0089','queued',NULL),
('d0090','queued','{"state":"pending"}'),
('d0091','queued','{"state":"determined","x":1}'),
('d0092','queued','{"state":"pending"}'),
('d0093','queued',NULL),
('d0094','queued','{"state":"pending"}'),
('d0095','queued','{"state":"determined","x":1}'),
('d0096','queued','{"state":"pending"}'),
('d0097','queued',NULL),
('d0098','queued','{"state":"pending"}'),
('d0099','queued','{"state":"determined","x":1}'),
('d0100','queued','{"state":"pending"}'),
('d0101','queued',NULL),
('d0102','queued','{"state":"pending"}'),
('d0103','queued','{"state":"determined","x":1}'),
('d0104','queued','{"state":"pending"}'),
('d0105','queued',NULL),
('d0106','queued','{"state":"pending"}'),
('d0107','queued','{"state":"determined","x":1}'),
('d0108','queued','{"state":"pending"}'),
('d0109','queued',NULL),
('d0110','queued','{"state":"pending"}'),
('d0111','queued','{"state":"determined","x":1}'),
('d0112','queued','{"state":"pending"}'),
('d0113','queued',NULL),
('d0114','queued','{"state":"pending"}'),
('d0115','queued','{"state":"determined","x":1}'),
('d0116','queued','{"state":"pending"}'),
('d0117','queued',NULL),
('d0118','queued','{"state":"pending"}'),
('d0119','queued','{"state":"determined","x":1}'),
('d0120','queued','{"state":"pending"}'),
('d0121','queued',NULL),
('d0122','queued','{"state":"pending"}'),
('d0123','queued','{"state":"determined","x":1}'),
('d0124','queued','{"state":"pending"}'),
('d0125','queued',NULL),
('d0126','queued','{"state":"pending"}'),
('d0127','queued','{"state":"determined","x":1}'),
('d0128','queued','{"state":"pending"}'),
('d0129','queued',NULL),
('d0130','queued','{"state":"pending"}'),
('d0131','queued','{"state":"determined","x":1}'),
('d0132','queued','{"state":"pending"}'),
('d0133','queued',NULL),
('d0134','queued','{"state":"pending"}'),
('d0135','queued','{"state":"determined","x":1}'),
('d0136','queued','{"state":"pending"}'),
('d0137','queued',NULL),
('d0138','queued','{"state":"pending"}'),
('d0139','queued','{"state":"determined","x":1}'),
('d0140','queued','{"state":"pending"}'),
('d0141','queued',NULL),
('d0142','queued','{"state":"pending"}'),
('d0143','queued','{"state":"determined","x":1}'),
('d0144','queued','{"state":"pending"}'),
('d0145','queued',NULL),
('d0146','queued','{"state":"pending"}'),
('d0147','queued','{"state":"determined","x":1}'),
('d0148','queued','{"state":"pending"}'),
('d0149','queued',NULL),
('d0150','queued','{"state":"pending"}'),
('d0151','queued','{"state":"determined","x":1}'),
('d0152','queued','{"state":"pending"}'),
('d0153','queued',NULL),
('d0154','queued','{"state":"pending"}'),
('d0155','queued','{"state":"determined","x":1}'),
('d0156','queued','{"state":"pending"}'),
('d0157','queued',NULL),
('d0158','queued','{"state":"pending"}'),
('d0159','queued','{"state":"determined","x":1}'),
('d0160','queued','{"state":"pending"}'),
('d0161','queued',NULL),
('d0162','queued','{"state":"pending"}'),
('d0163','queued','{"state":"determined","x":1}'),
('d0164','queued','{"state":"pending"}'),
('d0165','queued',NULL),
('d0166','queued','{"state":"pending"}'),
('d0167','queued','{"state":"determined","x":1}'),
('d0168','queued','{"state":"pending"}'),
('d0169','queued',NULL),
('d0170','queued','{"state":"pending"}'),
('d0171','queued','{"state":"determined","x":1}'),
('d0172','queued','{"state":"pending"}'),
('d0173','queued',NULL),
('d0174','queued','{"state":"pending"}'),
('d0175','queued','{"state":"determined","x":1}'),
('d0176','queued','{"state":"pending"}'),
('d0177','queued',NULL),
('d0178','queued','{"state":"pending"}'),
('d0179','queued','{"state":"determined","x":1}'),
('d0180','queued','{"state":"pending"}'),
('d0181','queued',NULL),
('d0182','queued','{"state":"pending"}'),
('d0183','queued','{"state":"determined","x":1}'),
('d0184','queued','{"state":"pending"}'),
('d0185','queued',NULL),
('d0186','queued','{"state":"pending"}'),
('d0187','queued','{"state":"determined","x":1}'),
('d0188','queued','{"state":"pending"}'),
('d0189','queued',NULL),
('d0190','queued','{"state":"pending"}'),
('d0191','queued','{"state":"determined","x":1}'),
('d0192','queued','{"state":"pending"}'),
('d0193','queued',NULL),
('d0194','queued','{"state":"pending"}'),
('d0195','queued','{"state":"determined","x":1}'),
('d0196','queued','{"state":"pending"}'),
('d0197','queued',NULL),
('d0198','queued','{"state":"pending"}'),
('d0199','queued','{"state":"determined","x":1}'),
('d0200','queued','{"state":"pending"}'),
('d0201','queued',NULL),
('d0202','queued','{"state":"pending"}'),
('d0203','queued','{"state":"determined","x":1}'),
('d0204','queued','{"state":"pending"}'),
('d0205','queued',NULL),
('d0206','queued','{"state":"pending"}'),
('d0207','queued','{"state":"determined","x":1}'),
('d0208','queued','{"state":"pending"}'),
('d0209','queued',NULL),
('d0210','queued','{"state":"pending"}'),
('d0211','queued','{"state":"determined","x":1}'),
('d0212','queued','{"state":"pending"}'),
('d0213','queued',NULL),
('d0214','queued','{"state":"pending"}'),
('d0215','queued','{"state":"determined","x":1}'),
('d0216','queued','{"state":"pending"}'),
('d0217','queued',NULL),
('d0218','queued','{"state":"pending"}'),
('d0219','queued','{"state":"determined","x":1}'),
('d0220','queued','{"state":"pending"}'),
('d0221','queued',NULL),
('d0222','queued','{"state":"pending"}'),
('d0223','queued','{"state":"determined","x":1}'),
('d0224','queued','{"state":"pending"}'),
('d0225','queued',NULL),
('d0226','queued','{"state":"pending"}'),
('d0227','queued','{"state":"determined","x":1}'),
('d0228','queued','{"state":"pending"}'),
('d0229','queued',NULL),
('d0230','queued','{"state":"pending"}'),
('d0231','queued','{"state":"determined","x":1}'),
('d0232','queued','{"state":"pending"}'),
('d0233','queued',NULL),
('d0234','queued','{"state":"pending"}'),
('d0235','queued','{"state":"determined","x":1}'),
('d0236','queued','{"state":"pending"}'),
('d0237','queued',NULL),
('d0238','queued','{"state":"pending"}'),
('d0239','queued','{"state":"determined","x":1}'),
('d0240','queued','{"state":"pending"}'),
('d0241','queued',NULL),
('d0242','queued','{"state":"pending"}'),
('d0243','queued','{"state":"determined","x":1}'),
('d0244','queued','{"state":"pending"}'),
('d0245','queued',NULL),
('d0246','queued','{"state":"pending"}'),
('d0247','queued','{"state":"determined","x":1}'),
('d0248','queued','{"state":"pending"}'),
('d0249','queued',NULL),
('d0250','queued','{"state":"pending"}'),
('d0251','queued','{"state":"determined","x":1}'),
('d0252','queued','{"state":"pending"}'),
('d0253','queued',NULL),
('d0254','queued','{"state":"pending"}'),
('d0255','queued','{"state":"determined","x":1}'),
('d0256','queued','{"state":"pending"}'),
('d0257','queued',NULL),
('d0258','queued','{"state":"pending"}'),
('d0259','queued','{"state":"determined","x":1}'),
('d0260','queued','{"state":"pending"}'),
('d0261','queued',NULL),
('d0262','queued','{"state":"pending"}'),
('d0263','queued','{"state":"determined","x":1}'),
('d0264','queued','{"state":"pending"}'),
('d0265','queued',NULL),
('d0266','queued','{"state":"pending"}'),
('d0267','queued','{"state":"determined","x":1}'),
('d0268','queued','{"state":"pending"}'),
('d0269','queued',NULL),
('d0270','queued','{"state":"pending"}'),
('d0271','queued','{"state":"determined","x":1}'),
('d0272','queued','{"state":"pending"}'),
('d0273','queued',NULL),
('d0274','queued','{"state":"pending"}'),
('d0275','queued','{"state":"determined","x":1}'),
('d0276','queued','{"state":"pending"}'),
('d0277','queued',NULL),
('d0278','queued','{"state":"pending"}'),
('d0279','queued','{"state":"determined","x":1}'),
('d0280','queued','{"state":"pending"}'),
('d0281','queued',NULL),
('d0282','queued','{"state":"pending"}'),
('d0283','queued','{"state":"determined","x":1}'),
('d0284','queued','{"state":"pending"}'),
('d0285','queued',NULL),
('d0286','queued','{"state":"pending"}'),
('d0287','queued','{"state":"determined","x":1}'),
('d0288','queued','{"state":"pending"}'),
('d0289','queued',NULL),
('d0290','queued','{"state":"pending"}'),
('d0291','queued','{"state":"determined","x":1}'),
('d0292','queued','{"state":"pending"}'),
('d0293','queued',NULL),
('d0294','queued','{"state":"pending"}'),
('d0295','queued','{"state":"determined","x":1}'),
('d0296','queued','{"state":"pending"}'),
('d0297','queued',NULL),
('d0298','queued','{"state":"pending"}'),
('d0299','queued','{"state":"determined","x":1}'),
('d0300','queued','{"state":"pending"}'),
('d0301','queued',NULL),
('d0302','queued','{"state":"pending"}'),
('d0303','queued','{"state":"determined","x":1}'),
('d0304','queued','{"state":"pending"}'),
('d0305','queued',NULL),
('d0306','queued','{"state":"pending"}'),
('d0307','queued','{"state":"determined","x":1}'),
('d0308','queued','{"state":"pending"}'),
('d0309','queued',NULL),
('d0310','queued','{"state":"pending"}'),
('d0311','queued','{"state":"determined","x":1}'),
('d0312','queued','{"state":"pending"}'),
('d0313','queued',NULL),
('d0314','queued','{"state":"pending"}'),
('d0315','queued','{"state":"determined","x":1}'),
('d0316','queued','{"state":"pending"}'),
('d0317','queued',NULL),
('d0318','queued','{"state":"pending"}'),
('d0319','queued','{"state":"determined","x":1}'),
('d0320','queued','{"state":"pending"}'),
('d0321','queued',NULL),
('d0322','queued','{"state":"pending"}'),
('d0323','queued','{"state":"determined","x":1}'),
('d0324','queued','{"state":"pending"}'),
('d0325','queued',NULL),
('d0326','queued','{"state":"pending"}'),
('d0327','queued','{"state":"determined","x":1}'),
('d0328','queued','{"state":"pending"}'),
('d0329','queued',NULL),
('d0330','queued','{"state":"pending"}'),
('d0331','queued','{"state":"determined","x":1}'),
('d0332','queued','{"state":"pending"}'),
('d0333','queued',NULL),
('d0334','queued','{"state":"pending"}'),
('d0335','queued','{"state":"determined","x":1}'),
('d0336','queued','{"state":"pending"}'),
('d0337','queued',NULL),
('d0338','queued','{"state":"pending"}'),
('d0339','queued','{"state":"determined","x":1}'),
('d0340','queued','{"state":"pending"}'),
('d0341','queued',NULL),
('d0342','queued','{"state":"pending"}'),
('d0343','queued','{"state":"determined","x":1}'),
('d0344','queued','{"state":"pending"}'),
('d0345','queued',NULL),
('d0346','queued','{"state":"pending"}'),
('d0347','queued','{"state":"determined","x":1}'),
('d0348','queued','{"state":"pending"}'),
('d0349','queued',NULL),
('d0350','queued','{"state":"pending"}'),
('d0351','queued','{"state":"determined","x":1}'),
('d0352','queued','{"state":"pending"}'),
('d0353','queued',NULL),
('d0354','queued','{"state":"pending"}'),
('d0355','queued','{"state":"determined","x":1}'),
('d0356','queued','{"state":"pending"}'),
('d0357','queued',NULL),
('d0358','queued','{"state":"pending"}'),
('d0359','queued','{"state":"determined","x":1}'),
('d0360','queued','{"state":"pending"}'),
('d0361','queued',NULL),
('d0362','queued','{"state":"pending"}'),
('d0363','queued','{"state":"determined","x":1}'),
('d0364','queued','{"state":"pending"}'),
('d0365','queued',NULL),
('d0366','queued','{"state":"pending"}'),
('d0367','queued','{"state":"determined","x":1}'),
('d0368','queued','{"state":"pending"}'),
('d0369','queued',NULL),
('d0370','queued','{"state":"pending"}'),
('d0371','queued','{"state":"determined","x":1}'),
('d0372','queued','{"state":"pending"}'),
('d0373','queued',NULL),
('d0374','queued','{"state":"pending"}'),
('d0375','queued','{"state":"determined","x":1}'),
('d0376','queued','{"state":"pending"}'),
('d0377','queued',NULL),
('d0378','queued','{"state":"pending"}'),
('d0379','queued','{"state":"determined","x":1}'),
('d0380','queued','{"state":"pending"}'),
('d0381','queued',NULL),
('d0382','queued','{"state":"pending"}'),
('d0383','queued','{"state":"determined","x":1}'),
('d0384','queued','{"state":"pending"}'),
('d0385','queued',NULL),
('d0386','queued','{"state":"pending"}'),
('d0387','queued','{"state":"determined","x":1}'),
('d0388','queued','{"state":"pending"}'),
('d0389','queued',NULL),
('d0390','queued','{"state":"pending"}'),
('d0391','queued','{"state":"determined","x":1}'),
('d0392','queued','{"state":"pending"}'),
('d0393','queued',NULL),
('d0394','queued','{"state":"pending"}'),
('d0395','queued','{"state":"determined","x":1}'),
('d0396','queued','{"state":"pending"}'),
('d0397','queued',NULL),
('d0398','queued','{"state":"pending"}'),
('d0399','queued','{"state":"determined","x":1}'),
('d0400','running','{"state":"pending"}'),
('d0401','running',NULL),
('d0402','running','{"state":"pending"}'),
('d0403','running','{"state":"determined","x":1}'),
('d0404','running','{"state":"pending"}'),
('d0405','running',NULL),
('d0406','running','{"state":"pending"}'),
('d0407','running','{"state":"determined","x":1}'),
('d0408','running','{"state":"pending"}'),
('d0409','running',NULL),
('d0410','running','{"state":"pending"}'),
('d0411','running','{"state":"determined","x":1}'),
('d0412','running','{"state":"pending"}'),
('d0413','running',NULL),
('d0414','running','{"state":"pending"}'),
('d0415','running','{"state":"determined","x":1}'),
('d0416','running','{"state":"pending"}'),
('d0417','running',NULL),
('d0418','running','{"state":"pending"}'),
('d0419','running','{"state":"determined","x":1}'),
('d0420','running','{"state":"pending"}'),
('d0421','running',NULL),
('d0422','running','{"state":"pending"}'),
('d0423','running','{"state":"determined","x":1}'),
('d0424','running','{"state":"pending"}'),
('d0425','running',NULL),
('d0426','running','{"state":"pending"}'),
('d0427','running','{"state":"determined","x":1}'),
('d0428','running','{"state":"pending"}'),
('d0429','running',NULL),
('d0430','running','{"state":"pending"}'),
('d0431','running','{"state":"determined","x":1}'),
('d0432','running','{"state":"pending"}'),
('d0433','running',NULL),
('d0434','running','{"state":"pending"}'),
('d0435','running','{"state":"determined","x":1}'),
('d0436','running','{"state":"pending"}'),
('d0437','running',NULL),
('d0438','running','{"state":"pending"}'),
('d0439','running','{"state":"determined","x":1}'),
('d0440','running','{"state":"pending"}'),
('d0441','running',NULL),
('d0442','running','{"state":"pending"}'),
('d0443','running','{"state":"determined","x":1}'),
('d0444','running','{"state":"pending"}'),
('d0445','running',NULL),
('d0446','running','{"state":"pending"}'),
('d0447','running','{"state":"determined","x":1}'),
('d0448','running','{"state":"pending"}'),
('d0449','running',NULL),
('d0450','failed','{"state":"pending"}'),
('d0451','failed','{"state":"determined","x":1}'),
('d0452','failed','{"state":"pending"}'),
('d0453','failed',NULL),
('d0454','failed','{"state":"pending"}'),
('d0455','failed','{"state":"determined","x":1}'),
('d0456','failed','{"state":"pending"}'),
('d0457','failed',NULL),
('d0458','failed','{"state":"pending"}'),
('d0459','failed','{"state":"determined","x":1}'),
('d0460','failed','{"state":"pending"}'),
('d0461','failed',NULL),
('d0462','failed','{"state":"pending"}'),
('d0463','failed','{"state":"determined","x":1}'),
('d0464','failed','{"state":"pending"}'),
('d0465','failed',NULL),
('d0466','failed','{"state":"pending"}'),
('d0467','failed','{"state":"determined","x":1}'),
('d0468','failed','{"state":"pending"}'),
('d0469','failed',NULL),
('d0470','failed','{"state":"pending"}'),
('d0471','failed','{"state":"determined","x":1}'),
('d0472','failed','{"state":"pending"}'),
('d0473','failed',NULL),
('d0474','failed','{"state":"pending"}'),
('d0475','failed','{"state":"determined","x":1}'),
('d0476','failed','{"state":"pending"}'),
('d0477','failed',NULL),
('d0478','failed','{"state":"pending"}'),
('d0479','failed','{"state":"determined","x":1}'),
('d0480','completed','{"state":"pending"}'),
('d0481','completed',NULL),
('d0482','completed','{"state":"pending"}'),
('d0483','completed','{"state":"determined","x":1}'),
('d0484','completed','{"state":"pending"}'),
('d0485','completed',NULL),
('d0486','completed','{"state":"pending"}'),
('d0487','completed','{"state":"determined","x":1}'),
('d0488','completed','{"state":"pending"}'),
('d0489','completed',NULL),
('d0490','completed','{"state":"pending"}'),
('d0491','completed','{"state":"determined","x":1}'),
('d0492','completed','{"state":"pending"}'),
('d0493','completed',NULL),
('d0494','completed','{"state":"pending"}'),
('d0495','completed','{"state":"determined","x":1}'),
('d0496','completed','{"state":"pending"}'),
('d0497','completed',NULL),
('d0498','completed','{"state":"pending"}'),
('d0499','completed','{"state":"determined","x":1}');
.parameter init
.parameter set $1 'd0000'
.parameter set $2 'dependency dep0 finished failed'
.parameter set $3 'd0001'
.parameter set $4 'dependency dep1 finished failed'
.parameter set $5 'd0002'
.parameter set $6 'dependency dep2 finished failed'
.parameter set $7 'd0003'
.parameter set $8 'dependency dep3 finished failed'
.parameter set $9 'd0004'
.parameter set $10 'dependency dep4 finished failed'
.parameter set $11 'd0005'
.parameter set $12 'dependency dep5 finished failed'
.parameter set $13 'd0006'
.parameter set $14 'dependency dep6 finished failed'
.parameter set $15 'd0007'
.parameter set $16 'dependency dep7 finished failed'
.parameter set $17 'd0008'
.parameter set $18 'dependency dep8 finished failed'
.parameter set $19 'd0009'
.parameter set $20 'dependency dep9 finished failed'
.parameter set $21 'd0010'
.parameter set $22 'dependency dep10 finished failed'
.parameter set $23 'd0011'
.parameter set $24 'dependency dep11 finished failed'
.parameter set $25 'd0012'
.parameter set $26 'dependency dep12 finished failed'
.parameter set $27 'd0013'
.parameter set $28 'dependency dep13 finished failed'
.parameter set $29 'd0014'
.parameter set $30 'dependency dep14 finished failed'
.parameter set $31 'd0015'
.parameter set $32 'dependency dep15 finished failed'
.parameter set $33 'd0016'
.parameter set $34 'dependency dep16 finished failed'
.parameter set $35 'd0017'
.parameter set $36 'dependency dep17 finished failed'
.parameter set $37 'd0018'
.parameter set $38 'dependency dep18 finished failed'
.parameter set $39 'd0019'
.parameter set $40 'dependency dep19 finished failed'
.parameter set $41 'd0020'
.parameter set $42 'dependency dep20 finished failed'
.parameter set $43 'd0021'
.parameter set $44 'dependency dep21 finished failed'
.parameter set $45 'd0022'
.parameter set $46 'dependency dep22 finished failed'
.parameter set $47 'd0023'
.parameter set $48 'dependency dep23 finished failed'
.parameter set $49 'd0024'
.parameter set $50 'dependency dep24 finished failed'
.parameter set $51 'd0025'
.parameter set $52 'dependency dep25 finished failed'
.parameter set $53 'd0026'
.parameter set $54 'dependency dep26 finished failed'
.parameter set $55 'd0027'
.parameter set $56 'dependency dep27 finished failed'
.parameter set $57 'd0028'
.parameter set $58 'dependency dep28 finished failed'
.parameter set $59 'd0029'
.parameter set $60 'dependency dep29 finished failed'
.parameter set $61 'd0030'
.parameter set $62 'dependency dep30 finished failed'
.parameter set $63 'd0031'
.parameter set $64 'dependency dep31 finished failed'
.parameter set $65 'd0032'
.parameter set $66 'dependency dep32 finished failed'
.parameter set $67 'd0033'
.parameter set $68 'dependency dep33 finished failed'
.parameter set $69 'd0034'
.parameter set $70 'dependency dep34 finished failed'
.parameter set $71 'd0035'
.parameter set $72 'dependency dep35 finished failed'
.parameter set $73 'd0036'
.parameter set $74 'dependency dep36 finished failed'
.parameter set $75 'd0037'
.parameter set $76 'dependency dep37 finished failed'
.parameter set $77 'd0038'
.parameter set $78 'dependency dep38 finished failed'
.parameter set $79 'd0039'
.parameter set $80 'dependency dep39 finished failed'
.parameter set $81 'd0040'
.parameter set $82 'dependency dep40 finished failed'
.parameter set $83 'd0041'
.parameter set $84 'dependency dep41 finished failed'
.parameter set $85 'd0042'
.parameter set $86 'dependency dep42 finished failed'
.parameter set $87 'd0043'
.parameter set $88 'dependency dep43 finished failed'
.parameter set $89 'd0044'
.parameter set $90 'dependency dep44 finished failed'
.parameter set $91 'd0045'
.parameter set $92 'dependency dep45 finished failed'
.parameter set $93 'd0046'
.parameter set $94 'dependency dep46 finished failed'
.parameter set $95 'd0047'
.parameter set $96 'dependency dep47 finished failed'
.parameter set $97 'd0048'
.parameter set $98 'dependency dep48 finished failed'
.parameter set $99 'd0049'
.parameter set $100 'dependency dep49 finished failed'
.parameter set $101 'd0050'
.parameter set $102 'dependency dep50 finished failed'
.parameter set $103 'd0051'
.parameter set $104 'dependency dep51 finished failed'
.parameter set $105 'd0052'
.parameter set $106 'dependency dep52 finished failed'
.parameter set $107 'd0053'
.parameter set $108 'dependency dep53 finished failed'
.parameter set $109 'd0054'
.parameter set $110 'dependency dep54 finished failed'
.parameter set $111 'd0055'
.parameter set $112 'dependency dep55 finished failed'
.parameter set $113 'd0056'
.parameter set $114 'dependency dep56 finished failed'
.parameter set $115 'd0057'
.parameter set $116 'dependency dep57 finished failed'
.parameter set $117 'd0058'
.parameter set $118 'dependency dep58 finished failed'
.parameter set $119 'd0059'
.parameter set $120 'dependency dep59 finished failed'
.parameter set $121 'd0060'
.parameter set $122 'dependency dep60 finished failed'
.parameter set $123 'd0061'
.parameter set $124 'dependency dep61 finished failed'
.parameter set $125 'd0062'
.parameter set $126 'dependency dep62 finished failed'
.parameter set $127 'd0063'
.parameter set $128 'dependency dep63 finished failed'
.parameter set $129 'd0064'
.parameter set $130 'dependency dep64 finished failed'
.parameter set $131 'd0065'
.parameter set $132 'dependency dep65 finished failed'
.parameter set $133 'd0066'
.parameter set $134 'dependency dep66 finished failed'
.parameter set $135 'd0067'
.parameter set $136 'dependency dep67 finished failed'
.parameter set $137 'd0068'
.parameter set $138 'dependency dep68 finished failed'
.parameter set $139 'd0069'
.parameter set $140 'dependency dep69 finished failed'
.parameter set $141 'd0070'
.parameter set $142 'dependency dep70 finished failed'
.parameter set $143 'd0071'
.parameter set $144 'dependency dep71 finished failed'
.parameter set $145 'd0072'
.parameter set $146 'dependency dep72 finished failed'
.parameter set $147 'd0073'
.parameter set $148 'dependency dep73 finished failed'
.parameter set $149 'd0074'
.parameter set $150 'dependency dep74 finished failed'
.parameter set $151 'd0075'
.parameter set $152 'dependency dep75 finished failed'
.parameter set $153 'd0076'
.parameter set $154 'dependency dep76 finished failed'
.parameter set $155 'd0077'
.parameter set $156 'dependency dep77 finished failed'
.parameter set $157 'd0078'
.parameter set $158 'dependency dep78 finished failed'
.parameter set $159 'd0079'
.parameter set $160 'dependency dep79 finished failed'
.parameter set $161 'd0080'
.parameter set $162 'dependency dep80 finished failed'
.parameter set $163 'd0081'
.parameter set $164 'dependency dep81 finished failed'
.parameter set $165 'd0082'
.parameter set $166 'dependency dep82 finished failed'
.parameter set $167 'd0083'
.parameter set $168 'dependency dep83 finished failed'
.parameter set $169 'd0084'
.parameter set $170 'dependency dep84 finished failed'
.parameter set $171 'd0085'
.parameter set $172 'dependency dep85 finished failed'
.parameter set $173 'd0086'
.parameter set $174 'dependency dep86 finished failed'
.parameter set $175 'd0087'
.parameter set $176 'dependency dep87 finished failed'
.parameter set $177 'd0088'
.parameter set $178 'dependency dep88 finished failed'
.parameter set $179 'd0089'
.parameter set $180 'dependency dep89 finished failed'
.parameter set $181 'd0090'
.parameter set $182 'dependency dep90 finished failed'
.parameter set $183 'd0091'
.parameter set $184 'dependency dep91 finished failed'
.parameter set $185 'd0092'
.parameter set $186 'dependency dep92 finished failed'
.parameter set $187 'd0093'
.parameter set $188 'dependency dep93 finished failed'
.parameter set $189 'd0094'
.parameter set $190 'dependency dep94 finished failed'
.parameter set $191 'd0095'
.parameter set $192 'dependency dep95 finished failed'
.parameter set $193 'd0096'
.parameter set $194 'dependency dep96 finished failed'
.parameter set $195 'd0097'
.parameter set $196 'dependency dep97 finished failed'
.parameter set $197 'd0098'
.parameter set $198 'dependency dep98 finished failed'
.parameter set $199 'd0099'
.parameter set $200 'dependency dep99 finished failed'
.parameter set $201 'd0100'
.parameter set $202 'dependency dep100 finished failed'
.parameter set $203 'd0101'
.parameter set $204 'dependency dep101 finished failed'
.parameter set $205 'd0102'
.parameter set $206 'dependency dep102 finished failed'
.parameter set $207 'd0103'
.parameter set $208 'dependency dep103 finished failed'
.parameter set $209 'd0104'
.parameter set $210 'dependency dep104 finished failed'
.parameter set $211 'd0105'
.parameter set $212 'dependency dep105 finished failed'
.parameter set $213 'd0106'
.parameter set $214 'dependency dep106 finished failed'
.parameter set $215 'd0107'
.parameter set $216 'dependency dep107 finished failed'
.parameter set $217 'd0108'
.parameter set $218 'dependency dep108 finished failed'
.parameter set $219 'd0109'
.parameter set $220 'dependency dep109 finished failed'
.parameter set $221 'd0110'
.parameter set $222 'dependency dep110 finished failed'
.parameter set $223 'd0111'
.parameter set $224 'dependency dep111 finished failed'
.parameter set $225 'd0112'
.parameter set $226 'dependency dep112 finished failed'
.parameter set $227 'd0113'
.parameter set $228 'dependency dep113 finished failed'
.parameter set $229 'd0114'
.parameter set $230 'dependency dep114 finished failed'
.parameter set $231 'd0115'
.parameter set $232 'dependency dep115 finished failed'
.parameter set $233 'd0116'
.parameter set $234 'dependency dep116 finished failed'
.parameter set $235 'd0117'
.parameter set $236 'dependency dep117 finished failed'
.parameter set $237 'd0118'
.parameter set $238 'dependency dep118 finished failed'
.parameter set $239 'd0119'
.parameter set $240 'dependency dep119 finished failed'
.parameter set $241 'd0120'
.parameter set $242 'dependency dep120 finished failed'
.parameter set $243 'd0121'
.parameter set $244 'dependency dep121 finished failed'
.parameter set $245 'd0122'
.parameter set $246 'dependency dep122 finished failed'
.parameter set $247 'd0123'
.parameter set $248 'dependency dep123 finished failed'
.parameter set $249 'd0124'
.parameter set $250 'dependency dep124 finished failed'
.parameter set $251 'd0125'
.parameter set $252 'dependency dep125 finished failed'
.parameter set $253 'd0126'
.parameter set $254 'dependency dep126 finished failed'
.parameter set $255 'd0127'
.parameter set $256 'dependency dep127 finished failed'
.parameter set $257 'd0128'
.parameter set $258 'dependency dep128 finished failed'
.parameter set $259 'd0129'
.parameter set $260 'dependency dep129 finished failed'
.parameter set $261 'd0130'
.parameter set $262 'dependency dep130 finished failed'
.parameter set $263 'd0131'
.parameter set $264 'dependency dep131 finished failed'
.parameter set $265 'd0132'
.parameter set $266 'dependency dep132 finished failed'
.parameter set $267 'd0133'
.parameter set $268 'dependency dep133 finished failed'
.parameter set $269 'd0134'
.parameter set $270 'dependency dep134 finished failed'
.parameter set $271 'd0135'
.parameter set $272 'dependency dep135 finished failed'
.parameter set $273 'd0136'
.parameter set $274 'dependency dep136 finished failed'
.parameter set $275 'd0137'
.parameter set $276 'dependency dep137 finished failed'
.parameter set $277 'd0138'
.parameter set $278 'dependency dep138 finished failed'
.parameter set $279 'd0139'
.parameter set $280 'dependency dep139 finished failed'
.parameter set $281 'd0140'
.parameter set $282 'dependency dep140 finished failed'
.parameter set $283 'd0141'
.parameter set $284 'dependency dep141 finished failed'
.parameter set $285 'd0142'
.parameter set $286 'dependency dep142 finished failed'
.parameter set $287 'd0143'
.parameter set $288 'dependency dep143 finished failed'
.parameter set $289 'd0144'
.parameter set $290 'dependency dep144 finished failed'
.parameter set $291 'd0145'
.parameter set $292 'dependency dep145 finished failed'
.parameter set $293 'd0146'
.parameter set $294 'dependency dep146 finished failed'
.parameter set $295 'd0147'
.parameter set $296 'dependency dep147 finished failed'
.parameter set $297 'd0148'
.parameter set $298 'dependency dep148 finished failed'
.parameter set $299 'd0149'
.parameter set $300 'dependency dep149 finished failed'
.parameter set $301 'd0150'
.parameter set $302 'dependency dep150 finished failed'
.parameter set $303 'd0151'
.parameter set $304 'dependency dep151 finished failed'
.parameter set $305 'd0152'
.parameter set $306 'dependency dep152 finished failed'
.parameter set $307 'd0153'
.parameter set $308 'dependency dep153 finished failed'
.parameter set $309 'd0154'
.parameter set $310 'dependency dep154 finished failed'
.parameter set $311 'd0155'
.parameter set $312 'dependency dep155 finished failed'
.parameter set $313 'd0156'
.parameter set $314 'dependency dep156 finished failed'
.parameter set $315 'd0157'
.parameter set $316 'dependency dep157 finished failed'
.parameter set $317 'd0158'
.parameter set $318 'dependency dep158 finished failed'
.parameter set $319 'd0159'
.parameter set $320 'dependency dep159 finished failed'
.parameter set $321 'd0160'
.parameter set $322 'dependency dep160 finished failed'
.parameter set $323 'd0161'
.parameter set $324 'dependency dep161 finished failed'
.parameter set $325 'd0162'
.parameter set $326 'dependency dep162 finished failed'
.parameter set $327 'd0163'
.parameter set $328 'dependency dep163 finished failed'
.parameter set $329 'd0164'
.parameter set $330 'dependency dep164 finished failed'
.parameter set $331 'd0165'
.parameter set $332 'dependency dep165 finished failed'
.parameter set $333 'd0166'
.parameter set $334 'dependency dep166 finished failed'
.parameter set $335 'd0167'
.parameter set $336 'dependency dep167 finished failed'
.parameter set $337 'd0168'
.parameter set $338 'dependency dep168 finished failed'
.parameter set $339 'd0169'
.parameter set $340 'dependency dep169 finished failed'
.parameter set $341 'd0170'
.parameter set $342 'dependency dep170 finished failed'
.parameter set $343 'd0171'
.parameter set $344 'dependency dep171 finished failed'
.parameter set $345 'd0172'
.parameter set $346 'dependency dep172 finished failed'
.parameter set $347 'd0173'
.parameter set $348 'dependency dep173 finished failed'
.parameter set $349 'd0174'
.parameter set $350 'dependency dep174 finished failed'
.parameter set $351 'd0175'
.parameter set $352 'dependency dep175 finished failed'
.parameter set $353 'd0176'
.parameter set $354 'dependency dep176 finished failed'
.parameter set $355 'd0177'
.parameter set $356 'dependency dep177 finished failed'
.parameter set $357 'd0178'
.parameter set $358 'dependency dep178 finished failed'
.parameter set $359 'd0179'
.parameter set $360 'dependency dep179 finished failed'
.parameter set $361 'd0180'
.parameter set $362 'dependency dep180 finished failed'
.parameter set $363 'd0181'
.parameter set $364 'dependency dep181 finished failed'
.parameter set $365 'd0182'
.parameter set $366 'dependency dep182 finished failed'
.parameter set $367 'd0183'
.parameter set $368 'dependency dep183 finished failed'
.parameter set $369 'd0184'
.parameter set $370 'dependency dep184 finished failed'
.parameter set $371 'd0185'
.parameter set $372 'dependency dep185 finished failed'
.parameter set $373 'd0186'
.parameter set $374 'dependency dep186 finished failed'
.parameter set $375 'd0187'
.parameter set $376 'dependency dep187 finished failed'
.parameter set $377 'd0188'
.parameter set $378 'dependency dep188 finished failed'
.parameter set $379 'd0189'
.parameter set $380 'dependency dep189 finished failed'
.parameter set $381 'd0190'
.parameter set $382 'dependency dep190 finished failed'
.parameter set $383 'd0191'
.parameter set $384 'dependency dep191 finished failed'
.parameter set $385 'd0192'
.parameter set $386 'dependency dep192 finished failed'
.parameter set $387 'd0193'
.parameter set $388 'dependency dep193 finished failed'
.parameter set $389 'd0194'
.parameter set $390 'dependency dep194 finished failed'
.parameter set $391 'd0195'
.parameter set $392 'dependency dep195 finished failed'
.parameter set $393 'd0196'
.parameter set $394 'dependency dep196 finished failed'
.parameter set $395 'd0197'
.parameter set $396 'dependency dep197 finished failed'
.parameter set $397 'd0198'
.parameter set $398 'dependency dep198 finished failed'
.parameter set $399 'd0199'
.parameter set $400 'dependency dep199 finished failed'
.parameter set $401 'd0200'
.parameter set $402 'dependency dep200 finished failed'
.parameter set $403 'd0201'
.parameter set $404 'dependency dep201 finished failed'
.parameter set $405 'd0202'
.parameter set $406 'dependency dep202 finished failed'
.parameter set $407 'd0203'
.parameter set $408 'dependency dep203 finished failed'
.parameter set $409 'd0204'
.parameter set $410 'dependency dep204 finished failed'
.parameter set $411 'd0205'
.parameter set $412 'dependency dep205 finished failed'
.parameter set $413 'd0206'
.parameter set $414 'dependency dep206 finished failed'
.parameter set $415 'd0207'
.parameter set $416 'dependency dep207 finished failed'
.parameter set $417 'd0208'
.parameter set $418 'dependency dep208 finished failed'
.parameter set $419 'd0209'
.parameter set $420 'dependency dep209 finished failed'
.parameter set $421 'd0210'
.parameter set $422 'dependency dep210 finished failed'
.parameter set $423 'd0211'
.parameter set $424 'dependency dep211 finished failed'
.parameter set $425 'd0212'
.parameter set $426 'dependency dep212 finished failed'
.parameter set $427 'd0213'
.parameter set $428 'dependency dep213 finished failed'
.parameter set $429 'd0214'
.parameter set $430 'dependency dep214 finished failed'
.parameter set $431 'd0215'
.parameter set $432 'dependency dep215 finished failed'
.parameter set $433 'd0216'
.parameter set $434 'dependency dep216 finished failed'
.parameter set $435 'd0217'
.parameter set $436 'dependency dep217 finished failed'
.parameter set $437 'd0218'
.parameter set $438 'dependency dep218 finished failed'
.parameter set $439 'd0219'
.parameter set $440 'dependency dep219 finished failed'
.parameter set $441 'd0220'
.parameter set $442 'dependency dep220 finished failed'
.parameter set $443 'd0221'
.parameter set $444 'dependency dep221 finished failed'
.parameter set $445 'd0222'
.parameter set $446 'dependency dep222 finished failed'
.parameter set $447 'd0223'
.parameter set $448 'dependency dep223 finished failed'
.parameter set $449 'd0224'
.parameter set $450 'dependency dep224 finished failed'
.parameter set $451 'd0225'
.parameter set $452 'dependency dep225 finished failed'
.parameter set $453 'd0226'
.parameter set $454 'dependency dep226 finished failed'
.parameter set $455 'd0227'
.parameter set $456 'dependency dep227 finished failed'
.parameter set $457 'd0228'
.parameter set $458 'dependency dep228 finished failed'
.parameter set $459 'd0229'
.parameter set $460 'dependency dep229 finished failed'
.parameter set $461 'd0230'
.parameter set $462 'dependency dep230 finished failed'
.parameter set $463 'd0231'
.parameter set $464 'dependency dep231 finished failed'
.parameter set $465 'd0232'
.parameter set $466 'dependency dep232 finished failed'
.parameter set $467 'd0233'
.parameter set $468 'dependency dep233 finished failed'
.parameter set $469 'd0234'
.parameter set $470 'dependency dep234 finished failed'
.parameter set $471 'd0235'
.parameter set $472 'dependency dep235 finished failed'
.parameter set $473 'd0236'
.parameter set $474 'dependency dep236 finished failed'
.parameter set $475 'd0237'
.parameter set $476 'dependency dep237 finished failed'
.parameter set $477 'd0238'
.parameter set $478 'dependency dep238 finished failed'
.parameter set $479 'd0239'
.parameter set $480 'dependency dep239 finished failed'
.parameter set $481 'd0240'
.parameter set $482 'dependency dep240 finished failed'
.parameter set $483 'd0241'
.parameter set $484 'dependency dep241 finished failed'
.parameter set $485 'd0242'
.parameter set $486 'dependency dep242 finished failed'
.parameter set $487 'd0243'
.parameter set $488 'dependency dep243 finished failed'
.parameter set $489 'd0244'
.parameter set $490 'dependency dep244 finished failed'
.parameter set $491 'd0245'
.parameter set $492 'dependency dep245 finished failed'
.parameter set $493 'd0246'
.parameter set $494 'dependency dep246 finished failed'
.parameter set $495 'd0247'
.parameter set $496 'dependency dep247 finished failed'
.parameter set $497 'd0248'
.parameter set $498 'dependency dep248 finished failed'
.parameter set $499 'd0249'
.parameter set $500 'dependency dep249 finished failed'
.parameter set $501 'd0250'
.parameter set $502 'dependency dep250 finished failed'
.parameter set $503 'd0251'
.parameter set $504 'dependency dep251 finished failed'
.parameter set $505 'd0252'
.parameter set $506 'dependency dep252 finished failed'
.parameter set $507 'd0253'
.parameter set $508 'dependency dep253 finished failed'
.parameter set $509 'd0254'
.parameter set $510 'dependency dep254 finished failed'
.parameter set $511 'd0255'
.parameter set $512 'dependency dep255 finished failed'
.parameter set $513 'd0256'
.parameter set $514 'dependency dep256 finished failed'
.parameter set $515 'd0257'
.parameter set $516 'dependency dep257 finished failed'
.parameter set $517 'd0258'
.parameter set $518 'dependency dep258 finished failed'
.parameter set $519 'd0259'
.parameter set $520 'dependency dep259 finished failed'
.parameter set $521 'd0260'
.parameter set $522 'dependency dep260 finished failed'
.parameter set $523 'd0261'
.parameter set $524 'dependency dep261 finished failed'
.parameter set $525 'd0262'
.parameter set $526 'dependency dep262 finished failed'
.parameter set $527 'd0263'
.parameter set $528 'dependency dep263 finished failed'
.parameter set $529 'd0264'
.parameter set $530 'dependency dep264 finished failed'
.parameter set $531 'd0265'
.parameter set $532 'dependency dep265 finished failed'
.parameter set $533 'd0266'
.parameter set $534 'dependency dep266 finished failed'
.parameter set $535 'd0267'
.parameter set $536 'dependency dep267 finished failed'
.parameter set $537 'd0268'
.parameter set $538 'dependency dep268 finished failed'
.parameter set $539 'd0269'
.parameter set $540 'dependency dep269 finished failed'
.parameter set $541 'd0270'
.parameter set $542 'dependency dep270 finished failed'
.parameter set $543 'd0271'
.parameter set $544 'dependency dep271 finished failed'
.parameter set $545 'd0272'
.parameter set $546 'dependency dep272 finished failed'
.parameter set $547 'd0273'
.parameter set $548 'dependency dep273 finished failed'
.parameter set $549 'd0274'
.parameter set $550 'dependency dep274 finished failed'
.parameter set $551 'd0275'
.parameter set $552 'dependency dep275 finished failed'
.parameter set $553 'd0276'
.parameter set $554 'dependency dep276 finished failed'
.parameter set $555 'd0277'
.parameter set $556 'dependency dep277 finished failed'
.parameter set $557 'd0278'
.parameter set $558 'dependency dep278 finished failed'
.parameter set $559 'd0279'
.parameter set $560 'dependency dep279 finished failed'
.parameter set $561 'd0280'
.parameter set $562 'dependency dep280 finished failed'
.parameter set $563 'd0281'
.parameter set $564 'dependency dep281 finished failed'
.parameter set $565 'd0282'
.parameter set $566 'dependency dep282 finished failed'
.parameter set $567 'd0283'
.parameter set $568 'dependency dep283 finished failed'
.parameter set $569 'd0284'
.parameter set $570 'dependency dep284 finished failed'
.parameter set $571 'd0285'
.parameter set $572 'dependency dep285 finished failed'
.parameter set $573 'd0286'
.parameter set $574 'dependency dep286 finished failed'
.parameter set $575 'd0287'
.parameter set $576 'dependency dep287 finished failed'
.parameter set $577 'd0288'
.parameter set $578 'dependency dep288 finished failed'
.parameter set $579 'd0289'
.parameter set $580 'dependency dep289 finished failed'
.parameter set $581 'd0290'
.parameter set $582 'dependency dep290 finished failed'
.parameter set $583 'd0291'
.parameter set $584 'dependency dep291 finished failed'
.parameter set $585 'd0292'
.parameter set $586 'dependency dep292 finished failed'
.parameter set $587 'd0293'
.parameter set $588 'dependency dep293 finished failed'
.parameter set $589 'd0294'
.parameter set $590 'dependency dep294 finished failed'
.parameter set $591 'd0295'
.parameter set $592 'dependency dep295 finished failed'
.parameter set $593 'd0296'
.parameter set $594 'dependency dep296 finished failed'
.parameter set $595 'd0297'
.parameter set $596 'dependency dep297 finished failed'
.parameter set $597 'd0298'
.parameter set $598 'dependency dep298 finished failed'
.parameter set $599 'd0299'
.parameter set $600 'dependency dep299 finished failed'
.parameter set $601 'd0300'
.parameter set $602 'dependency dep300 finished failed'
.parameter set $603 'd0301'
.parameter set $604 'dependency dep301 finished failed'
.parameter set $605 'd0302'
.parameter set $606 'dependency dep302 finished failed'
.parameter set $607 'd0303'
.parameter set $608 'dependency dep303 finished failed'
.parameter set $609 'd0304'
.parameter set $610 'dependency dep304 finished failed'
.parameter set $611 'd0305'
.parameter set $612 'dependency dep305 finished failed'
.parameter set $613 'd0306'
.parameter set $614 'dependency dep306 finished failed'
.parameter set $615 'd0307'
.parameter set $616 'dependency dep307 finished failed'
.parameter set $617 'd0308'
.parameter set $618 'dependency dep308 finished failed'
.parameter set $619 'd0309'
.parameter set $620 'dependency dep309 finished failed'
.parameter set $621 'd0310'
.parameter set $622 'dependency dep310 finished failed'
.parameter set $623 'd0311'
.parameter set $624 'dependency dep311 finished failed'
.parameter set $625 'd0312'
.parameter set $626 'dependency dep312 finished failed'
.parameter set $627 'd0313'
.parameter set $628 'dependency dep313 finished failed'
.parameter set $629 'd0314'
.parameter set $630 'dependency dep314 finished failed'
.parameter set $631 'd0315'
.parameter set $632 'dependency dep315 finished failed'
.parameter set $633 'd0316'
.parameter set $634 'dependency dep316 finished failed'
.parameter set $635 'd0317'
.parameter set $636 'dependency dep317 finished failed'
.parameter set $637 'd0318'
.parameter set $638 'dependency dep318 finished failed'
.parameter set $639 'd0319'
.parameter set $640 'dependency dep319 finished failed'
.parameter set $641 'd0320'
.parameter set $642 'dependency dep320 finished failed'
.parameter set $643 'd0321'
.parameter set $644 'dependency dep321 finished failed'
.parameter set $645 'd0322'
.parameter set $646 'dependency dep322 finished failed'
.parameter set $647 'd0323'
.parameter set $648 'dependency dep323 finished failed'
.parameter set $649 'd0324'
.parameter set $650 'dependency dep324 finished failed'
.parameter set $651 'd0325'
.parameter set $652 'dependency dep325 finished failed'
.parameter set $653 'd0326'
.parameter set $654 'dependency dep326 finished failed'
.parameter set $655 'd0327'
.parameter set $656 'dependency dep327 finished failed'
.parameter set $657 'd0328'
.parameter set $658 'dependency dep328 finished failed'
.parameter set $659 'd0329'
.parameter set $660 'dependency dep329 finished failed'
.parameter set $661 'd0330'
.parameter set $662 'dependency dep330 finished failed'
.parameter set $663 'd0331'
.parameter set $664 'dependency dep331 finished failed'
.parameter set $665 'd0332'
.parameter set $666 'dependency dep332 finished failed'
.parameter set $667 'd0333'
.parameter set $668 'dependency dep333 finished failed'
.parameter set $669 'd0334'
.parameter set $670 'dependency dep334 finished failed'
.parameter set $671 'd0335'
.parameter set $672 'dependency dep335 finished failed'
.parameter set $673 'd0336'
.parameter set $674 'dependency dep336 finished failed'
.parameter set $675 'd0337'
.parameter set $676 'dependency dep337 finished failed'
.parameter set $677 'd0338'
.parameter set $678 'dependency dep338 finished failed'
.parameter set $679 'd0339'
.parameter set $680 'dependency dep339 finished failed'
.parameter set $681 'd0340'
.parameter set $682 'dependency dep340 finished failed'
.parameter set $683 'd0341'
.parameter set $684 'dependency dep341 finished failed'
.parameter set $685 'd0342'
.parameter set $686 'dependency dep342 finished failed'
.parameter set $687 'd0343'
.parameter set $688 'dependency dep343 finished failed'
.parameter set $689 'd0344'
.parameter set $690 'dependency dep344 finished failed'
.parameter set $691 'd0345'
.parameter set $692 'dependency dep345 finished failed'
.parameter set $693 'd0346'
.parameter set $694 'dependency dep346 finished failed'
.parameter set $695 'd0347'
.parameter set $696 'dependency dep347 finished failed'
.parameter set $697 'd0348'
.parameter set $698 'dependency dep348 finished failed'
.parameter set $699 'd0349'
.parameter set $700 'dependency dep349 finished failed'
.parameter set $701 'd0350'
.parameter set $702 'dependency dep350 finished failed'
.parameter set $703 'd0351'
.parameter set $704 'dependency dep351 finished failed'
.parameter set $705 'd0352'
.parameter set $706 'dependency dep352 finished failed'
.parameter set $707 'd0353'
.parameter set $708 'dependency dep353 finished failed'
.parameter set $709 'd0354'
.parameter set $710 'dependency dep354 finished failed'
.parameter set $711 'd0355'
.parameter set $712 'dependency dep355 finished failed'
.parameter set $713 'd0356'
.parameter set $714 'dependency dep356 finished failed'
.parameter set $715 'd0357'
.parameter set $716 'dependency dep357 finished failed'
.parameter set $717 'd0358'
.parameter set $718 'dependency dep358 finished failed'
.parameter set $719 'd0359'
.parameter set $720 'dependency dep359 finished failed'
.parameter set $721 'd0360'
.parameter set $722 'dependency dep360 finished failed'
.parameter set $723 'd0361'
.parameter set $724 'dependency dep361 finished failed'
.parameter set $725 'd0362'
.parameter set $726 'dependency dep362 finished failed'
.parameter set $727 'd0363'
.parameter set $728 'dependency dep363 finished failed'
.parameter set $729 'd0364'
.parameter set $730 'dependency dep364 finished failed'
.parameter set $731 'd0365'
.parameter set $732 'dependency dep365 finished failed'
.parameter set $733 'd0366'
.parameter set $734 'dependency dep366 finished failed'
.parameter set $735 'd0367'
.parameter set $736 'dependency dep367 finished failed'
.parameter set $737 'd0368'
.parameter set $738 'dependency dep368 finished failed'
.parameter set $739 'd0369'
.parameter set $740 'dependency dep369 finished failed'
.parameter set $741 'd0370'
.parameter set $742 'dependency dep370 finished failed'
.parameter set $743 'd0371'
.parameter set $744 'dependency dep371 finished failed'
.parameter set $745 'd0372'
.parameter set $746 'dependency dep372 finished failed'
.parameter set $747 'd0373'
.parameter set $748 'dependency dep373 finished failed'
.parameter set $749 'd0374'
.parameter set $750 'dependency dep374 finished failed'
.parameter set $751 'd0375'
.parameter set $752 'dependency dep375 finished failed'
.parameter set $753 'd0376'
.parameter set $754 'dependency dep376 finished failed'
.parameter set $755 'd0377'
.parameter set $756 'dependency dep377 finished failed'
.parameter set $757 'd0378'
.parameter set $758 'dependency dep378 finished failed'
.parameter set $759 'd0379'
.parameter set $760 'dependency dep379 finished failed'
.parameter set $761 'd0380'
.parameter set $762 'dependency dep380 finished failed'
.parameter set $763 'd0381'
.parameter set $764 'dependency dep381 finished failed'
.parameter set $765 'd0382'
.parameter set $766 'dependency dep382 finished failed'
.parameter set $767 'd0383'
.parameter set $768 'dependency dep383 finished failed'
.parameter set $769 'd0384'
.parameter set $770 'dependency dep384 finished failed'
.parameter set $771 'd0385'
.parameter set $772 'dependency dep385 finished failed'
.parameter set $773 'd0386'
.parameter set $774 'dependency dep386 finished failed'
.parameter set $775 'd0387'
.parameter set $776 'dependency dep387 finished failed'
.parameter set $777 'd0388'
.parameter set $778 'dependency dep388 finished failed'
.parameter set $779 'd0389'
.parameter set $780 'dependency dep389 finished failed'
.parameter set $781 'd0390'
.parameter set $782 'dependency dep390 finished failed'
.parameter set $783 'd0391'
.parameter set $784 'dependency dep391 finished failed'
.parameter set $785 'd0392'
.parameter set $786 'dependency dep392 finished failed'
.parameter set $787 'd0393'
.parameter set $788 'dependency dep393 finished failed'
.parameter set $789 'd0394'
.parameter set $790 'dependency dep394 finished failed'
.parameter set $791 'd0395'
.parameter set $792 'dependency dep395 finished failed'
.parameter set $793 'd0396'
.parameter set $794 'dependency dep396 finished failed'
.parameter set $795 'd0397'
.parameter set $796 'dependency dep397 finished failed'
.parameter set $797 'd0398'
.parameter set $798 'dependency dep398 finished failed'
.parameter set $799 'd0399'
.parameter set $800 'dependency dep399 finished failed'
.parameter set $801 'd0400'
.parameter set $802 'dependency dep400 finished failed'
.parameter set $803 'd0401'
.parameter set $804 'dependency dep401 finished failed'
.parameter set $805 'd0402'
.parameter set $806 'dependency dep402 finished failed'
.parameter set $807 'd0403'
.parameter set $808 'dependency dep403 finished failed'
.parameter set $809 'd0404'
.parameter set $810 'dependency dep404 finished failed'
.parameter set $811 'd0405'
.parameter set $812 'dependency dep405 finished failed'
.parameter set $813 'd0406'
.parameter set $814 'dependency dep406 finished failed'
.parameter set $815 'd0407'
.parameter set $816 'dependency dep407 finished failed'
.parameter set $817 'd0408'
.parameter set $818 'dependency dep408 finished failed'
.parameter set $819 'd0409'
.parameter set $820 'dependency dep409 finished failed'
.parameter set $821 'd0410'
.parameter set $822 'dependency dep410 finished failed'
.parameter set $823 'd0411'
.parameter set $824 'dependency dep411 finished failed'
.parameter set $825 'd0412'
.parameter set $826 'dependency dep412 finished failed'
.parameter set $827 'd0413'
.parameter set $828 'dependency dep413 finished failed'
.parameter set $829 'd0414'
.parameter set $830 'dependency dep414 finished failed'
.parameter set $831 'd0415'
.parameter set $832 'dependency dep415 finished failed'
.parameter set $833 'd0416'
.parameter set $834 'dependency dep416 finished failed'
.parameter set $835 'd0417'
.parameter set $836 'dependency dep417 finished failed'
.parameter set $837 'd0418'
.parameter set $838 'dependency dep418 finished failed'
.parameter set $839 'd0419'
.parameter set $840 'dependency dep419 finished failed'
.parameter set $841 'd0420'
.parameter set $842 'dependency dep420 finished failed'
.parameter set $843 'd0421'
.parameter set $844 'dependency dep421 finished failed'
.parameter set $845 'd0422'
.parameter set $846 'dependency dep422 finished failed'
.parameter set $847 'd0423'
.parameter set $848 'dependency dep423 finished failed'
.parameter set $849 'd0424'
.parameter set $850 'dependency dep424 finished failed'
.parameter set $851 'd0425'
.parameter set $852 'dependency dep425 finished failed'
.parameter set $853 'd0426'
.parameter set $854 'dependency dep426 finished failed'
.parameter set $855 'd0427'
.parameter set $856 'dependency dep427 finished failed'
.parameter set $857 'd0428'
.parameter set $858 'dependency dep428 finished failed'
.parameter set $859 'd0429'
.parameter set $860 'dependency dep429 finished failed'
.parameter set $861 'd0430'
.parameter set $862 'dependency dep430 finished failed'
.parameter set $863 'd0431'
.parameter set $864 'dependency dep431 finished failed'
.parameter set $865 'd0432'
.parameter set $866 'dependency dep432 finished failed'
.parameter set $867 'd0433'
.parameter set $868 'dependency dep433 finished failed'
.parameter set $869 'd0434'
.parameter set $870 'dependency dep434 finished failed'
.parameter set $871 'd0435'
.parameter set $872 'dependency dep435 finished failed'
.parameter set $873 'd0436'
.parameter set $874 'dependency dep436 finished failed'
.parameter set $875 'd0437'
.parameter set $876 'dependency dep437 finished failed'
.parameter set $877 'd0438'
.parameter set $878 'dependency dep438 finished failed'
.parameter set $879 'd0439'
.parameter set $880 'dependency dep439 finished failed'
.parameter set $881 'd0440'
.parameter set $882 'dependency dep440 finished failed'
.parameter set $883 'd0441'
.parameter set $884 'dependency dep441 finished failed'
.parameter set $885 'd0442'
.parameter set $886 'dependency dep442 finished failed'
.parameter set $887 'd0443'
.parameter set $888 'dependency dep443 finished failed'
.parameter set $889 'd0444'
.parameter set $890 'dependency dep444 finished failed'
.parameter set $891 'd0445'
.parameter set $892 'dependency dep445 finished failed'
.parameter set $893 'd0446'
.parameter set $894 'dependency dep446 finished failed'
.parameter set $895 'd0447'
.parameter set $896 'dependency dep447 finished failed'
.parameter set $897 'd0448'
.parameter set $898 'dependency dep448 finished failed'
.parameter set $899 'd0449'
.parameter set $900 'dependency dep449 finished failed'
.parameter set $901 'd0450'
.parameter set $902 'dependency dep450 finished failed'
.parameter set $903 'd0451'
.parameter set $904 'dependency dep451 finished failed'
.parameter set $905 'd0452'
.parameter set $906 'dependency dep452 finished failed'
.parameter set $907 'd0453'
.parameter set $908 'dependency dep453 finished failed'
.parameter set $909 'd0454'
.parameter set $910 'dependency dep454 finished failed'
.parameter set $911 'd0455'
.parameter set $912 'dependency dep455 finished failed'
.parameter set $913 'd0456'
.parameter set $914 'dependency dep456 finished failed'
.parameter set $915 'd0457'
.parameter set $916 'dependency dep457 finished failed'
.parameter set $917 'd0458'
.parameter set $918 'dependency dep458 finished failed'
.parameter set $919 'd0459'
.parameter set $920 'dependency dep459 finished failed'
.parameter set $921 'd0460'
.parameter set $922 'dependency dep460 finished failed'
.parameter set $923 'd0461'
.parameter set $924 'dependency dep461 finished failed'
.parameter set $925 'd0462'
.parameter set $926 'dependency dep462 finished failed'
.parameter set $927 'd0463'
.parameter set $928 'dependency dep463 finished failed'
.parameter set $929 'd0464'
.parameter set $930 'dependency dep464 finished failed'
.parameter set $931 'd0465'
.parameter set $932 'dependency dep465 finished failed'
.parameter set $933 'd0466'
.parameter set $934 'dependency dep466 finished failed'
.parameter set $935 'd0467'
.parameter set $936 'dependency dep467 finished failed'
.parameter set $937 'd0468'
.parameter set $938 'dependency dep468 finished failed'
.parameter set $939 'd0469'
.parameter set $940 'dependency dep469 finished failed'
.parameter set $941 'd0470'
.parameter set $942 'dependency dep470 finished failed'
.parameter set $943 'd0471'
.parameter set $944 'dependency dep471 finished failed'
.parameter set $945 'd0472'
.parameter set $946 'dependency dep472 finished failed'
.parameter set $947 'd0473'
.parameter set $948 'dependency dep473 finished failed'
.parameter set $949 'd0474'
.parameter set $950 'dependency dep474 finished failed'
.parameter set $951 'd0475'
.parameter set $952 'dependency dep475 finished failed'
.parameter set $953 'd0476'
.parameter set $954 'dependency dep476 finished failed'
.parameter set $955 'd0477'
.parameter set $956 'dependency dep477 finished failed'
.parameter set $957 'd0478'
.parameter set $958 'dependency dep478 finished failed'
.parameter set $959 'd0479'
.parameter set $960 'dependency dep479 finished failed'
.parameter set $961 'd0480'
.parameter set $962 'dependency dep480 finished failed'
.parameter set $963 'd0481'
.parameter set $964 'dependency dep481 finished failed'
.parameter set $965 'd0482'
.parameter set $966 'dependency dep482 finished failed'
.parameter set $967 'd0483'
.parameter set $968 'dependency dep483 finished failed'
.parameter set $969 'd0484'
.parameter set $970 'dependency dep484 finished failed'
.parameter set $971 'd0485'
.parameter set $972 'dependency dep485 finished failed'
.parameter set $973 'd0486'
.parameter set $974 'dependency dep486 finished failed'
.parameter set $975 'd0487'
.parameter set $976 'dependency dep487 finished failed'
.parameter set $977 'd0488'
.parameter set $978 'dependency dep488 finished failed'
.parameter set $979 'd0489'
.parameter set $980 'dependency dep489 finished failed'
.parameter set $981 'd0490'
.parameter set $982 'dependency dep490 finished failed'
.parameter set $983 'd0491'
.parameter set $984 'dependency dep491 finished failed'
.parameter set $985 'd0492'
.parameter set $986 'dependency dep492 finished failed'
.parameter set $987 'd0493'
.parameter set $988 'dependency dep493 finished failed'
.parameter set $989 'd0494'
.parameter set $990 'dependency dep494 finished failed'
.parameter set $991 'd0495'
.parameter set $992 'dependency dep495 finished failed'
.parameter set $993 'd0496'
.parameter set $994 'dependency dep496 finished failed'
.parameter set $995 'd0497'
.parameter set $996 'dependency dep497 finished failed'
.parameter set $997 'd0498'
.parameter set $998 'dependency dep498 finished failed'
.parameter set $999 'd0499'
.parameter set $1000 'dependency dep499 finished failed'
.parameter set $1001 '{"state":"pending"}'
.parameter set $1002 '{"state":"undetermined","reason":"dependency_failed"}'
.parameter set $1003 '2026-09-10T00:00:00Z'
.mode list
SELECT 'MARK_RETURNING';
WITH v(id,msg) AS (VALUES ($1,$2),($3,$4),($5,$6),($7,$8),($9,$10),($11,$12),($13,$14),($15,$16),($17,$18),($19,$20),($21,$22),($23,$24),($25,$26),($27,$28),($29,$30),($31,$32),($33,$34),($35,$36),($37,$38),($39,$40),($41,$42),($43,$44),($45,$46),($47,$48),($49,$50),($51,$52),($53,$54),($55,$56),($57,$58),($59,$60),($61,$62),($63,$64),($65,$66),($67,$68),($69,$70),($71,$72),($73,$74),($75,$76),($77,$78),($79,$80),($81,$82),($83,$84),($85,$86),($87,$88),($89,$90),($91,$92),($93,$94),($95,$96),($97,$98),($99,$100),($101,$102),($103,$104),($105,$106),($107,$108),($109,$110),($111,$112),($113,$114),($115,$116),($117,$118),($119,$120),($121,$122),($123,$124),($125,$126),($127,$128),($129,$130),($131,$132),($133,$134),($135,$136),($137,$138),($139,$140),($141,$142),($143,$144),($145,$146),($147,$148),($149,$150),($151,$152),($153,$154),($155,$156),($157,$158),($159,$160),($161,$162),($163,$164),($165,$166),($167,$168),($169,$170),($171,$172),($173,$174),($175,$176),($177,$178),($179,$180),($181,$182),($183,$184),($185,$186),($187,$188),($189,$190),($191,$192),($193,$194),($195,$196),($197,$198),($199,$200),($201,$202),($203,$204),($205,$206),($207,$208),($209,$210),($211,$212),($213,$214),($215,$216),($217,$218),($219,$220),($221,$222),($223,$224),($225,$226),($227,$228),($229,$230),($231,$232),($233,$234),($235,$236),($237,$238),($239,$240),($241,$242),($243,$244),($245,$246),($247,$248),($249,$250),($251,$252),($253,$254),($255,$256),($257,$258),($259,$260),($261,$262),($263,$264),($265,$266),($267,$268),($269,$270),($271,$272),($273,$274),($275,$276),($277,$278),($279,$280),($281,$282),($283,$284),($285,$286),($287,$288),($289,$290),($291,$292),($293,$294),($295,$296),($297,$298),($299,$300),($301,$302),($303,$304),($305,$306),($307,$308),($309,$310),($311,$312),($313,$314),($315,$316),($317,$318),($319,$320),($321,$322),($323,$324),($325,$326),($327,$328),($329,$330),($331,$332),($333,$334),($335,$336),($337,$338),($339,$340),($341,$342),($343,$344),($345,$346),($347,$348),($349,$350),($351,$352),($353,$354),($355,$356),($357,$358),($359,$360),($361,$362),($363,$364),($365,$366),($367,$368),($369,$370),($371,$372),($373,$374),($375,$376),($377,$378),($379,$380),($381,$382),($383,$384),($385,$386),($387,$388),($389,$390),($391,$392),($393,$394),($395,$396),($397,$398),($399,$400),($401,$402),($403,$404),($405,$406),($407,$408),($409,$410),($411,$412),($413,$414),($415,$416),($417,$418),($419,$420),($421,$422),($423,$424),($425,$426),($427,$428),($429,$430),($431,$432),($433,$434),($435,$436),($437,$438),($439,$440),($441,$442),($443,$444),($445,$446),($447,$448),($449,$450),($451,$452),($453,$454),($455,$456),($457,$458),($459,$460),($461,$462),($463,$464),($465,$466),($467,$468),($469,$470),($471,$472),($473,$474),($475,$476),($477,$478),($479,$480),($481,$482),($483,$484),($485,$486),($487,$488),($489,$490),($491,$492),($493,$494),($495,$496),($497,$498),($499,$500),($501,$502),($503,$504),($505,$506),($507,$508),($509,$510),($511,$512),($513,$514),($515,$516),($517,$518),($519,$520),($521,$522),($523,$524),($525,$526),($527,$528),($529,$530),($531,$532),($533,$534),($535,$536),($537,$538),($539,$540),($541,$542),($543,$544),($545,$546),($547,$548),($549,$550),($551,$552),($553,$554),($555,$556),($557,$558),($559,$560),($561,$562),($563,$564),($565,$566),($567,$568),($569,$570),($571,$572),($573,$574),($575,$576),($577,$578),($579,$580),($581,$582),($583,$584),($585,$586),($587,$588),($589,$590),($591,$592),($593,$594),($595,$596),($597,$598),($599,$600),($601,$602),($603,$604),($605,$606),($607,$608),($609,$610),($611,$612),($613,$614),($615,$616),($617,$618),($619,$620),($621,$622),($623,$624),($625,$626),($627,$628),($629,$630),($631,$632),($633,$634),($635,$636),($637,$638),($639,$640),($641,$642),($643,$644),($645,$646),($647,$648),($649,$650),($651,$652),($653,$654),($655,$656),($657,$658),($659,$660),($661,$662),($663,$664),($665,$666),($667,$668),($669,$670),($671,$672),($673,$674),($675,$676),($677,$678),($679,$680),($681,$682),($683,$684),($685,$686),($687,$688),($689,$690),($691,$692),($693,$694),($695,$696),($697,$698),($699,$700),($701,$702),($703,$704),($705,$706),($707,$708),($709,$710),($711,$712),($713,$714),($715,$716),($717,$718),($719,$720),($721,$722),($723,$724),($725,$726),($727,$728),($729,$730),($731,$732),($733,$734),($735,$736),($737,$738),($739,$740),($741,$742),($743,$744),($745,$746),($747,$748),($749,$750),($751,$752),($753,$754),($755,$756),($757,$758),($759,$760),($761,$762),($763,$764),($765,$766),($767,$768),($769,$770),($771,$772),($773,$774),($775,$776),($777,$778),($779,$780),($781,$782),($783,$784),($785,$786),($787,$788),($789,$790),($791,$792),($793,$794),($795,$796),($797,$798),($799,$800),($801,$802),($803,$804),($805,$806),($807,$808),($809,$810),($811,$812),($813,$814),($815,$816),($817,$818),($819,$820),($821,$822),($823,$824),($825,$826),($827,$828),($829,$830),($831,$832),($833,$834),($835,$836),($837,$838),($839,$840),($841,$842),($843,$844),($845,$846),($847,$848),($849,$850),($851,$852),($853,$854),($855,$856),($857,$858),($859,$860),($861,$862),($863,$864),($865,$866),($867,$868),($869,$870),($871,$872),($873,$874),($875,$876),($877,$878),($879,$880),($881,$882),($883,$884),($885,$886),($887,$888),($889,$890),($891,$892),($893,$894),($895,$896),($897,$898),($899,$900),($901,$902),($903,$904),($905,$906),($907,$908),($909,$910),($911,$912),($913,$914),($915,$916),($917,$918),($919,$920),($921,$922),($923,$924),($925,$926),($927,$928),($929,$930),($931,$932),($933,$934),($935,$936),($937,$938),($939,$940),($941,$942),($943,$944),($945,$946),($947,$948),($949,$950),($951,$952),($953,$954),($955,$956),($957,$958),($959,$960),($961,$962),($963,$964),($965,$966),($967,$968),($969,$970),($971,$972),($973,$974),($975,$976),($977,$978),($979,$980),($981,$982),($983,$984),($985,$986),($987,$988),($989,$990),($991,$992),($993,$994),($995,$996),($997,$998),($999,$1000)) UPDATE jobs SET status = 'failed', error = v.msg, acceleration_report = CASE WHEN acceleration_report = $1001 THEN $1002 ELSE acceleration_report END, updated_at = $1003 FROM v WHERE jobs.job_id = v.id AND jobs.status = 'queued' RETURNING jobs.job_id;
SELECT 'MARK_ASSERTS';
SELECT 'status_hist', status, count(*) FROM jobs GROUP BY 1,2 ORDER BY 2;
SELECT 'errmsg_wrong_among_touched', count(*) FROM jobs j WHERE j.updated_at<>'' AND j.error IS NOT 'dependency dep'||CAST(substr(j.job_id,2) AS INTEGER)||' finished failed';
SELECT 'touched_count', count(*) FROM jobs WHERE updated_at<>'';
SELECT 'ar_of_touched', coalesce(acceleration_report,'<NULL>'), count(*) FROM jobs WHERE updated_at<>'' GROUP BY 2 ORDER BY 2;
SELECT 'ar_of_untouched', coalesce(acceleration_report,'<NULL>'), count(*) FROM jobs WHERE updated_at='' GROUP BY 2 ORDER BY 2;
