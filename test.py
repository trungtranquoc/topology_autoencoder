import json

class MyClass:
    def __init__(self, name, age, city):
        self.name = name
        self.age = age
        self.city = city

    def save_to_json(self, filename):
        data_to_save = {k: v for k, v in self.__dict__.items() if k not in ["city"]}
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4) # indent for readability
    
    @staticmethod
    def load_from_json(filename):
        class_obj = MyClass(None, None, "None")
        with open(filename, 'r') as f:
            data = json.load(f)
            class_obj.__dict__.update(data)
        return class_obj

class_obj = MyClass("John", 30, "New York")
class_obj.save_to_json("class_obj.json")


class_obj_2 = MyClass.load_from_json("class_obj.json")
print(class_obj_2.name, class_obj_2.age, class_obj_2.city)