FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

# install some needed system utilities
# locales to set system encoding to utf-8
# openssh-server to ssh into container
RUN apt-get update && apt-get install -y locales openssh-server

# set system encoding to be UTF-8
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# set up openssh server
# from https://docs.docker.com/engine/examples/running_ssh_service/
RUN mkdir /var/run/sshd
# note the root password you set!
RUN echo 'root:abc123' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# install fastai
RUN conda install -y -c fastai fastai && \
    conda install -y jupyter notebook && \
    conda install -y -c conda-forge jupyter_contrib_nbextensions

# install other libraries
RUN conda install -y scikit-learn pyarrow pytest

# set the default anaconda environment
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# expose port 8888 in case you want to connect to a jupyter notebook
EXPOSE 8888
EXPOSE 22

WORKDIR /opt/rossmann
COPY . .

CMD ["/usr/sbin/sshd", "-D"]