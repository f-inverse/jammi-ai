INSERT INTO jobs(job_id,status,has_deps) VALUES
 ('r0001','failed',0),
 ('b0001','queued',1),('b0002','queued',1),
 ('c0001','queued',1),('c0002','queued',1),('c0003','queued',1),('c0004','queued',1),
 ('m0001','queued',1),('m0002','queued',1),
 ('k0001','completed',0),('x0001','queued',1),('y0001','queued',1),('z0001','queued',0);
INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES
 ('b0001','r0001'),('b0002','r0001'),
 ('c0001','b0001'),('c0002','b0001'),('c0003','b0002'),('c0004','b0002'),
 ('m0002','b0002'),('m0002','b0001'),
 ('m0001','b0001'),('m0001','b0002'),
 ('x0001','k0001'),
 ('y0001','b0001'),('y0001','k0001');
