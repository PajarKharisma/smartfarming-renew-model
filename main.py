import os
import torch

def get_solution(dir_solution):
    results = []

    files = os.listdir(dir_solution)
    for file in files:
        full_path = os.path.join(dir_solution,file)
        with open(full_path) as reader:
            result = reader.read().replace('\n','').split('-')
            result = [i for i in result if i]
            results.append(result)

    return results

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# checkpoint = torch.load('model-850a84c89d914d99a984619ba3fe1338.pth', map_location=device)
checkpoint = torch.load('smart-farming-model.pth', map_location=device)
# for sol in checkpoint['solution']:
#     print(sol)
solutions = get_solution('solution')
print(len(solutions))
for i in range(len(solutions)):
    print(solutions[i][0])
    print()
# checkpoint = {
#         'title' : checkpoint['title'],
#         'epoch': checkpoint['epoch'],
#         'loss' : checkpoint['loss'],
#         'class_names' : checkpoint['class_names'],
#         'desc' : checkpoint['desc'],
#         'solution' : solutions,
#         'state_dict': checkpoint['state_dict'],
#         'optimizer': checkpoint['optimizer']
#     }
# torch.save(checkpoint, 'smart-farming-model.pth')