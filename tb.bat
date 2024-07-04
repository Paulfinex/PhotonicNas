@echo off
cd /d %~dp0
tensorboard --logdir=./lightning_logs
pause