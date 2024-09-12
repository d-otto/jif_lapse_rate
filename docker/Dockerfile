# DockerFile for a firedrake + jupyter container

FROM icepack/firedrake-python3.10:0.6.1

USER sermilik
WORKDIR /home/sermilik

# PETSC_ARCH inherited from firedrakeproject/firedrake
# Environment required for Azure deployments.
ENV OMPI_MCA_btl tcp,self
ENV PATH /home/firedrake/firedrake/bin:$PATH

# install other packages
# todo: make this point to requirements.txt
RUN bash -c "python3 -m pip install dill"
RUN bash -c "python3 -m pip install geopandas shapely rasterio meshpy"
RUN bash -c "python3 -m pip install numpy pandas scipy scikit-learn statsmodels"
RUN bash -c "python3 -m pip install matplotlib seaborn cartopy"
RUN bash -c "python3 -m pip install xarray metpy xdem xarray-spatial rioxarray"

CMD ["/home/firedrake/firedrake/bin/jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]]
