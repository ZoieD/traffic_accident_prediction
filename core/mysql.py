import pymysql as pm

class MYSQL():
    def __init__(self,host,user,pwd,db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db

    def __GetConnect(self):
        """ Connect mssql
        """
        if not self.db:
            raise(NameError,"No database information is set")
        self.conn = pm.connect(host=self.host,user=self.user,password=self.pwd,database=self.db)
        cur = self.conn.cursor()
        if not cur:
            print("Unable to connect to the database")
        else:
            return cur

    def ExecQuery(self,sql):
        """ Execlude query
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()

        #The connection must be closed after the query is completed
        self.conn.close()
        return resList

    def ExecNonQuery(self,sql):
        """ Execlude non query
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()