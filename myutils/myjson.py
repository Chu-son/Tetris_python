#! /usr/bin/env python
#-*- coding:utf-8 -*-

import json


class JsonAdapter(object):
    """
        reference:http://qiita.com/Thiru0000/items/35554f523565e4b12b51
    """
    # jsonからシリアライズ
    def serialize(self, argJsonData) :

        jsonData = argJsonData
        if isinstance(argJsonData, str ) :
            print("loads JSON data")
            jsonData = json.loads(argJsonData)

        for key in self.__dict__.keys():
            if jsonData[key] == "NotSerializable":
                continue
            elif isinstance(getattr(self,key), JsonAdapter ) or getattr(self,key) == None :
                valclass = getattr(self,key)
                print("よくわからないやつ")
                #if valclass == None :
                #    valclass = getattr(sys.modules["Classes.DataPDO"],key)()

                valclass.serialize(jsonData[key])
                setattr(self,key, valclass)

            elif isinstance(getattr(self,key), int ) :
                setattr(self,key, int(jsonData[key]))
            elif isinstance(jsonData[key], (list, tuple, dict)):
                for item in jsonData[key]:
                    if item == "NotSerializable":
                        break
                else:
                    setattr(self,key, jsonData[key])
            else :
                setattr(self,key, jsonData[key])

    # Jsonを出力
    def to_json(self):
        jsonDict = self.__to_dictionary()
        jsonstring = json.dumps(jsonDict, ensure_ascii=False, default = self.__type_error, indent = 4)
        return jsonstring

    #Mapを作成
    def __to_dictionary(self):
        jsonDict = {}
        for key in self.__dict__.keys():
            if isinstance(getattr(self,key), JsonAdapter ) :
                jsonDict.update({key : getattr(self,key).__to_dictionary()})
            else :
                jsonDict.update({key : getattr(self,key)})
        return jsonDict

    def __type_error(self, obj):
        return str("NotSerializable")
