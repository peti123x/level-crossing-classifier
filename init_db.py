from peewee import *

db = SqliteDatabase('trains.db')

class Train(Model):
    id = AutoField()
    origin = CharField()
    destination = CharField()
    due = TimeField()
    delay = IntegerField()
    est = TimeField()
    cancelled = BooleanField()
    day = CharField()

    class Meta:
        database = db

db.connect()
db.create_tables([Train])

#train = Train(origin="LNC", destination="Sheffield", due="17:49", delay=0, est="17:49", cancelled=False,day="monday")
#success = train.save()
#print(success)

#mytrain = Train.get(Train.destination == "Sheffield")
#mytrain.delay = 3
#mytrain.est = "17:33"
#mytrain.save()

for train in Train.select().where(Train.destination == "Sheffield").order_by(Train.due.desc()).limit(1):
    train.est = "19:58"
    success = train.save()
    print(success)

db.close()
