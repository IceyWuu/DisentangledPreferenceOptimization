## 每次提交或修改代码之前需要做

git 基础流程

<img src="https://i-blog.csdnimg.cn/blog_migrate/48c5e7811d9ccd7f6a07442153c10df6.png" alt="在这里插入图片描述" style="zoom:25%;" />



**每天修改之前**：先拉取远程仓库中最新版本，确保在最新版本上进行修改。

```cmd
# 这是一键拉取并合并远程仓库中最新代码到本地，相当于执行了 git fetch + git rebase。
git pull --rebase origin main
```

**每次推送之前**：

暂存修改：如果本地有还没 commit 的代码，先commit。只提交重要内容。

```cmd
git status # 检查一下自己改了什么文件，有没有误改，如果有系统生成的垃圾文件，加到gitignore里。
git add xxx # xxx这里加上刚刚git status 检查过的，要提交的正确文件，那些.sh文件或者yaml文件，如果只是改了路径，或者调了不必要的参数（如跑哪个loss）之类的修改，就不要放上来了，避免一直需要处理冲突

git commit -m "..."

git stash # 如果存在没有commit的修改
```

然后拉取最新的补丁，有冲突就解决

```cmd
git pull --rebase origin main
```

没有问题就推送到main分支

```cmd
git push origin main
```

把没有commit的修改放回来，如果想直接丢弃则直接 `git stash drop`即可

```cmd
git stash apply # 如果没有stash，这两句就都不要执行了
git stash drop
```

**拉取分支可能遇到的情况**

如果本地有修改了，又需要pull把远程最新代码更新过来

```cmd
git stash # 先暂存自己的代码
git pull --rebase origin main # 然后拉取最新的补丁，有冲突就解决

git stash apply # 有冲突需要解决
git stash drop
```

**关于解决冲突**

如果有冲突，Git 会停下来。需要**打开文件手动解决冲突**，然后

```cmd
git add <冲突的文件>
git rebase --continue
```



## git部分

### git基础

分支贡献度统计（不是主分支的贡献度不会显现在主页，需要额外设置）：[Git：分支与主分支不同的github贡献度|极客教程](https://geek-docs.com/git/git-questions/616_git_github_contributions_for_branch_different_than_master.html)



**切换branch之后，本地代码也会变成当前branch的代码**

[git创建远程仓库并上传代码到远程仓库中 - 心如止水~ - 博客园 (cnblogs.com)](https://www.cnblogs.com/ygfsy/p/13921592.html#:~:text=git创建远程仓库并上传代码到远程仓库中 第一步：我们需要先创建一个本地的版本库（其实也就是一个文件夹）。 你可以直接右击新建文件夹，也可以右击打开Git,bash命令行窗口通过命令来创建。 现在我通过命令行在桌面新建一个TEST文件夹（你也可以在其他任何地方创建这个文件夹），并且进入这个文件夹 第二步：通过命令git init把这个文件夹变成Git可管理的仓库)

[【git】- 将本地项目关联到github远程仓库 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/88246764#:~:text=1 如果本地项目已经通过git管理： 直接通过git remote add origin (仓库地址) 关联,(仓库地址) %3D> git push -u origin master (推送到远程github))

[Github 文件夹出现箭头，并且打不开文件_githubdesktopush时同时显示上下箭头-CSDN博客](https://blog.csdn.net/cxliebtdich/article/details/121623850#:~:text=当你在自己的项目里clone了别人的项目，github就将他视为一个子系统模块，导致在上传代码时该文件夹上传失败，并在github上显示向右的白色箭头。 解决：删除子文件夹里面的.git文件 执行git rm,--cached [文件夹名] 执行git add.)

[Git篇：使用Git将代码库更新到本地（完整版）_git如何更新本地项目-CSDN博客](https://blog.csdn.net/qq_15020543/article/details/85062650#:~:text=环境：Git已安装（皮一下）新建一文件夹右键，Git Bash Heregit init添加库git remote add origin 想要更新的源码地址,将库里的代码下载到本地git pull origin master 到此为止，第一次下载项目流程就结束了，之后进行更新不需要重复以上步骤配置更新地址git remote add up..._git如何更新本地项目)

### git回滚

回滚[关于Unstaged changes after reset的理解-CSDN博客](https://blog.csdn.net/a1030260075/article/details/129761382)

![在这里插入图片描述](markdown-img/GitCommand.assets/7f719d39541949c3bef28d8b16657944.jpeg)

查看历史版本号：`git log`

### 切换分支需要stash

[git合并指定commit——git合并某个特定的提交commit到指定的分支上_git merge 指定commit-CSDN博客](https://blog.csdn.net/weixin_44867717/article/details/120885717)

**场景**：现在的Bug你还没有解决，而上边又给你派了一个新的Bug，而这个Bug相比较现在正在苦思冥想的Bug比较容易解决。你想先解决新的Bug，可是之前的Bug还没有解决完而不能提交。怎么办?

**问题**：当我们正常使用Git切换分支时，会出现以下提示：

```bash
Please commit your changes or stash them before you switch branches.
```

使用 **`git stash`**将当前分支存起来（可以看到id号），**`git stash list`** 查看“存储”的列表，此时可以继续切换分支。任务完成后，`git stash apply` 命令恢复，此时储藏项目还会在列表中，需要用 `git stash drop` 来删除。

如果有一个分支上多个 stash，如果需要恢复指定的 stash ，可以在命令尾部加id，如 **`git stash apply stash@{0}`**，同样删除指定 stash 项目则执行如 **`git stash drop stash@{1}`** 。

用 `git stash pop` 命令，恢复的同时把 stash 存储列表的内容也删了。



**stash 恢复时出现冲突**：

[[Git\]执行git stash pop时的冲突解决_git stash pop 冲突-CSDN博客](https://blog.csdn.net/jy692405180/article/details/78520251)



### 将其他分支合并到当前分支

**1. 只想合并other-branch上的某些特定提交(commit)**

**场景**：在A分支上提交了一个commit，B分支也同样需要这个commit的代码。

* 在当前A分支（deploy/t），通过`git log`先找到A分支的commit代号（**简略ID**-29d9493d-前8位数）

* 执行以下命令 ,切换到B分支（deploy/pre）,通过`git cherry-pick`+`简略ID`,进行合并指定的commit提交记录

```bash
git checkout B
git cherry-pick 29d9493d(修改为自己的id)
git push (此处需要指定分支推送)
```

**2. 合并other-branch上的所有提交**

* 首先，确保你在 new-branch 上：

  ```bash
  git checkout new-branch
  ```

* 然后，使用 `git merge` 命令将 other-branch 上的更改合并到 new-branch 上：

  ```
  git merge other-branch
  ```

  这样，other-branch 上的所有提交都会被合并到 new-branch 上了。

**3. 想把branch A 里的某些文件提交到branch B中**

* **切换到 B branch**： 首先，确保你在 `B branch` 上：

```bash
git checkout B
```

* **从 A branch 检出你想要的文件**： 使用 `git checkout` 命令从 `A branch` 中检出特定文件：

```bash
git checkout A -- path/to/your/file1 path/to/your/file2
例如：
git checkout A -- file1.txt file2.txt
```

* **添加这些文件到 B branch 的暂存区**： 使用 `git add` 命令将这些文件添加到暂存区：

```bash
git add path/to/your/file1 path/to/your/file2
```

* **提交这些更改**： 使用 `git commit` 提交这些更改到 `B branch`：

```bash
git commit -m "Add specific files from A branch"
```

通过这些步骤，你就可以将 `A branch` 中某个 commit 的部分文件提交到 `B branch` 了。

**4. 想更精细地控制哪些提交会被合并**

* 首先，确保你在new-branch上：

```bash
git checkout new-branch
```

* 然后，启动交互式rebase：

```bash
git rebase -i other-branch
```

这将打开一个文本编辑器，列出所有other-branch相对于 new-branch 的提交。你可以选择保留、删除或修改这些提交。



### git忽略__ pycache __和 .pyc文件

[Git 为什么 Git 忽略不了 __pycache__ 文件夹|极客笔记](https://deepinout.com/git/git-questions/29_git_why_does_git_ignore_not_work_for___pycache___folder.html)

Git 无法忽略 **pycache** 文件夹的原因是，该文件夹下的 .pyc 文件已经被 Git 跟踪过。当我们将 **pycache** 文件夹添加到 .gitignore 文件中后，Git 不会自动从仓库中删除已经跟踪的文件。这就意味着一旦 **pycache** 文件夹中的 .pyc 文件被跟踪，即使后来将此文件夹添加到 .gitignore 中，Git 仍然会显示该文件夹的更改，并将其包含在版本控制中。



要解决 Git 无法忽略 **pycache** 文件夹的问题，我们需要执行以下几个步骤：

1. **清除缓存**

首先，我们需要清除 Git 缓存中已经跟踪的 **pycache** 文件夹及其下的 .pyc 文件。可以使用以下命令完成此操作：

```bash
$ git rm -r --cached __pycache__
$ git rm -r --cached *.pyc
```

这会将 **pycache** 文件夹和所有 .pyc 文件从 Git 缓存中移除，但不会删除实际的文件。

2. **提交变更**

接下来，我们需要提交变更以使清除的文件生效。可以使用以下命令提交变更：

```bash
$ git commit -m "Remove __pycache__ from Git tracking"
```

3. **删除已跟踪的文件**

完成提交后，我们需要手动删除已跟踪的 **pycache** 文件夹及其下的 .pyc 文件，以确保不再被 Git 跟踪。可以使用以下命令删除已跟踪的文件：

```bash
$ git rm -r --force __pycache__
$ git rm -r --force *.pyc
```

4. **更新 .gitignore 文件**

最后，我们需要更新 .gitignore 文件以确保 Git 不再跟踪 **pycache** 文件夹及其下的任何文件或子文件夹。可以编辑 .gitignore 文件并添加以下行：

```python
__pycache__/
*.pyc
```



### colab

#### colab ssh

[vscode连接Google colab - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/620439869)

[设置本地VS Code (PyCharm)+Google Colab (免费)GPU的机器学习环境 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/376881331)



#### colab github project

[使用Colab运行github中的项目_github在线运行项目用的gpu是谁的-CSDN博客](https://blog.csdn.net/LiuLongLeg/article/details/118150983)



## conda

conda换源

`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

若要还原，可用下面的命令：

`pip config unset global.index-url`

用.yaml创建conda环境：

`conda env create -f base.yml`

用.yaml更新conda环境：

`conda activate myenv
conda env update --file local.yml`
