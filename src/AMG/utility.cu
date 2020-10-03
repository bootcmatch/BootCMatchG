#pragma once

#define DELIM "%"

char linebuffer[BUFSIZE+1];

int    int_get_fp(FILE *fp)
{
  int temp;
  char *token;
  fgets(linebuffer,BUFSIZE,fp);
  token = strtok(linebuffer,DELIM);
  sscanf(token,"%d",&temp);
  return(temp);
}

double    double_get_fp(FILE *fp)
{
  double temp;
  char *token;
  fgets(linebuffer,BUFSIZE,fp);
  token = strtok(linebuffer,DELIM);
  sscanf(token,"%lf",&temp);
  return(temp);
}

char* string_get_fp(FILE *fp)
{
  char *token1, *token2;
  fgets(linebuffer,BUFSIZE,fp);
  token1 = strtok(linebuffer,DELIM);
  token2 = strtok(token1," ");
  return(strdup(token2));
}
