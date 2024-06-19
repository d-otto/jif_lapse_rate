# DockerFile for a firedrake + jupyter container

FROM icepack/firedrake-python3.10:0.6.1

USER sermilik
WORKDIR /home/sermilik

# PETSC_ARCH inherited from firedrakeproject/firedrake

# Install an iPython kernel for firedrake
RUN bash -c ". /home/firedrake/firedrake/bin/activate && pip install nbclassic nbformat nbconvert ipympl"
RUN bash -c ". /home/firedrake/firedrake/bin/activate && jupyter nbclassic --generate-config"

# Remove the install log.
RUN bash -c "rm firedrake-*"
# Put the notebooks in the working directory for the notebook server.
RUN bash -c "cp -r firedrake/src/firedrake/docs/notebooks/* ."
# Strip the output from the notebooks.
RUN bash -c '. /home/firedrake/firedrake/bin/activate && for file in *.ipynb; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $file; done'

# Environment required for Azure deployments.
ENV OMPI_MCA_btl tcp,self
ENV PATH /home/firedrake/firedrake/bin:$PATH

# install other packages
# todo: make this point to requirements.txt
RUN bash -c ". /home/firedrake/firedrake/bin/activate && pip install geopandas xdem shapely"

CMD ["/home/firedrake/firedrake/bin/jupyter", "nbclassic", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]