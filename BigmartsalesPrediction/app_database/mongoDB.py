import logging
import pymongo
import datetime
connString = "mongodb+srv://ml_projcet:ZS5fy8x5aisdtJ6G@cluster0.wzb80.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
# mongodb+srv://ml_projcet:<password>@cluster0.wzb80.mongodb.net/?retryWrites=true&w=majority

class MongoHandler(logging.Handler):
    """
    A logging handler that will record messages to a (optionally capped)
    MongoDB collection.
    >>> connection = pymongo.Connection()
    >>> collection = connection.db.log
    >>> logger = logging.getLogger("mongotest")
    >>> logger.addHandler(MongoHandler(drop=True))
    >>> logger.error("Hello, world!")
    >>> collection.find_one()['message']
    u'Hello, world!'
    """

    def __init__(self, level=logging.NOTSET,
                 database='test_project', collection='log', capped=True, size=100000,
                 drop=False):
        logging.Handler.__init__(self, level)
        self.connection = pymongo.MongoClient(connString)
        self.database = self.connection[database]

        if collection in self.database.list_collection_names():
            if drop:
                self.database.drop_collection(collection)
                self.collection = self.database.create_collection(
                    collection, {'capped': capped, 'size': size})
            else:
                self.collection = self.database[collection]
        else:
            self.collection = self.database.create_collection(
                collection, {'capped': capped, 'size': size})

    def emit(self, record):
        self.collection.save({'when': datetime.datetime.now(),
                              'levelno': record.levelno,
                              'levelname': record.levelname,
                              'message': record.msg})


if __name__ == '__main__':
    import doctest
    doctest.testmod()