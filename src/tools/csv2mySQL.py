#!/usr/bin/python

"""This script imports a csv file to a table that it creates
Usage:
csv2mySQL.py FILE DATABASE TABLENAME 'column description' [IGNORE_LINES] 
"""
# By Maarten

import MySQLdb, sys, os
import time, datetime

from warnings import filterwarnings
filterwarnings('ignore', category = MySQLdb.Warning)


DISABLE_KEYS = True
# DISABLE_KEYS = False

if len(sys.argv) != 5 and len(sys.argv) != 6:
    sys.stderr.write("""Check your arguments!
Usage:       csv2mySQL.py FILE DATABASE TABLENAME '(mysql column description)' [IGNORELINES]
Example:     csv2mySQL.py example.csv my_database my_new_table '(id int(10), name varchar(20))' 1
""")
    sys.exit(1)

start = time.time()
filename = sys.argv[1]
database = sys.argv[2]
table = sys.argv[3]
column_description = sys.argv[4]
ignore_lines = sys.argv[5] if len(sys.argv) == 6 else 0
files = [filename]

if os.path.isdir(filename):
    print "Found a directory, reading ALL files in %s" % filename
    files = [os.path.abspath(os.path.join(filename,f)) for f in os.listdir(filename) if os.path.isfile(os.path.join(filename,f))]
#sys.exit()

#db = MySQLdb.connect(db=database, local_infile = 1,read_default_file='~/.my.cnf', use_unicode=True, charset="utf8mb4")
db = MySQLdb.connect(db=database, local_infile = 1,read_default_file='~/.my.cnf', use_unicode=True, charset="utf8")
cur = db.cursor()

cur.execute("show tables like '%s'" % table)
tablesss = [item[0] for item in cur.fetchall()]
append = False
if len(tablesss) > 0:
    print "A table by that name already exists in the database... Do you wish to overwrite it? (y/n) (enter 'a' for appending to existing table)"
    answer = sys.stdin.readline()[0]
    if answer.lower() == 'a':
        append = True
    elif answer.lower() != 'y':
        print "Try again with a different table name"
        sys.exit(1)    
if not append:
    cur.execute("drop table if exists %s" % table)
    print "create table %s %s" % (table, column_description)
    cur.execute("create table %s %s" % (table, column_description))
else:
    print "appending to table [%s]" % table

if DISABLE_KEYS:
    print "alter table %s disable keys" % table
    cur.execute("alter table %s disable keys" % table)

print "importing data, reading %d file%s" % (len(files),"s" if len(files) > 1 else "")

for i, f in enumerate(files):
    # num_lines = sum(1 for line in open(f))
    cur.execute("""load data local infile '%s' into table %s fields terminated by ',' enclosed by '"' lines terminated by '\\n' ignore %s lines""" % (f, table, ignore_lines))
    if len(files) != 1 and (i+1) % 10 == 0:
        print "imported %d files out of %d..." % (i+1, len(files))
print "imported %d files out of %d..." % (len(files), len(files))
print "Your %s been imported!\nDatabase: %s, Table: %s" % ("files have" if len(files) > 1 else "file has",database,table)

if DISABLE_KEYS:
    print "alter table %s enable keys" % table
    cur.execute("alter table %s enable keys" % table)
print "Took %s to import" % (datetime.timedelta(seconds=(time.time()-start))) 
db.commit()
